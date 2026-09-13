"""
Transcriptor de Audios con Deepgram nova-3
------------------------------------------
Envía llamadas de audio (.wav) a Deepgram API para transcripción con
diarización de hablantes. Gestiona caché, reintentos y orquesta el
pipeline de procesamiento delegando el parseo a diarization_parser.

Arquitectura de 3 capas:
1. data/raw_transcripts/{call_id}.json (Caché e idempotencia de Deepgram)
2. data/transcripts/{call_id}.md       (Diálogo formateado por turnos)
3. data/processed/transcripts_summary.csv / .parquet (Metadatos analíticos)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from tqdm import tqdm
import pandas as pd

from diarization_parser import process_and_save_dialogue


DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIOS_DIR = BASE_DIR / "audios"
HUMAN_DIR = AUDIOS_DIR / "audios_humanos_censurados"
AI_DIR = AUDIOS_DIR / "audios_ia_censurados"

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_transcripts"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PROCESSED_DIR = DATA_DIR / "processed"


def load_api_key() -> str:
    """
    Carga la API key de Deepgram usando os.environ.
    Si no está exportada en el entorno, intenta cargarla desde el archivo .env local.
    """
    key = os.environ.get("DEEPGRAM_API_KEY")
    if key and key.strip():
        return key.strip()

    env_file = BASE_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPGRAM_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        os.environ["DEEPGRAM_API_KEY"] = val
                        return val

    secret_file = BASE_DIR / "deepgram_secret.txt"
    if secret_file.exists():
        val = secret_file.read_text(encoding="utf-8").strip()
        if val:
            os.environ["DEEPGRAM_API_KEY"] = val
            return val

    raise ValueError(
        "No se encontró DEEPGRAM_API_KEY en las variables de entorno (os.environ) ni en el archivo .env"
    )


def setup_directories() -> None:
    """Crea los directorios de almacenamiento en capas si no existen."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def transcribe_audio_file(
    client: httpx.Client,
    file_path: Path,
    call_id: str,
    api_key: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Envía un archivo de audio a Deepgram API con el modelo nova-3 y diarización activada.
    Implementa reintentos con backoff exponencial.
    """
    cache_path = RAW_DIR / f"{call_id}.json"

    # Capa 1: Retornar de caché si ya existe y es válido
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "results" in data:
                    return data
        except Exception:
            pass

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav"
    }
    params = {
        "model": "nova-3",
        "language": "es",
        "diarize": "true",
        "smart_format": "true",
        "utterances": "true"
    }

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.post(
                DEEPGRAM_API_URL,
                params=params,
                headers=headers,
                content=audio_bytes,
                timeout=120.0
            )
            if response.status_code == 200:
                result = response.json()
                with open(cache_path, "w", encoding="utf-8") as f_out:
                    json.dump(result, f_out, ensure_ascii=False, indent=2)
                return result
            elif response.status_code == 429:
                wait_sec = attempt * 5
                time.sleep(wait_sec)
            else:
                last_err = f"Status {response.status_code}: {response.text}"
                time.sleep(2)
        except Exception as e:
            last_err = str(e)
            time.sleep(attempt * 2)

    raise RuntimeError(f"Fallo al transcribir {file_path.name} ({call_id}): {last_err}")


def run_pipeline(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo para todos los audios con nomenclatura estandarizada:
    humano_01 .. humano_50 y ia_01 .. ia_50.
    Aprovecha la caché de Deepgram existente en data/raw_transcripts/ para no volver a gastar saldo.
    """
    setup_directories()
    api_key = load_api_key()

    human_files = sorted(list(HUMAN_DIR.glob("*.wav")))
    ai_files = sorted(list(AI_DIR.glob("*.wav")))

    if limit:
        human_files = human_files[:limit]
        ai_files = ai_files[:limit]

    human_tasks = [(f, "humano", f"humano_{idx + 1:02d}") for idx, f in enumerate(human_files)]
    ai_tasks = [(f, "ia", f"ia_{idx + 1:02d}") for idx, f in enumerate(ai_files)]
    all_tasks = human_tasks + ai_tasks

    print(f"Total de audios a procesar: {len(all_tasks)} (Humanos: {len(human_files)}, IA: {len(ai_files)})")

    summary_records = []
    with httpx.Client() as client:
        for file_path, source, call_id in tqdm(all_tasks, desc="Procesando y clasificando con Deepgram nova-3"):
            try:
                raw_json = transcribe_audio_file(client, file_path, call_id, api_key)
                meta = process_and_save_dialogue(
                    raw_json, call_id, file_path.name, source, TRANSCRIPTS_DIR
                )
                summary_records.append(meta)
            except Exception as err:
                print(f"Error procesando {file_path.name} ({call_id}): {err}", file=sys.stderr)

    df_summary = pd.DataFrame(summary_records)

    # Capa 3: Guardar metadatos analíticos en parquet y csv
    csv_path = PROCESSED_DIR / "transcripts_summary.csv"
    parquet_path = PROCESSED_DIR / "transcripts_summary.parquet"

    df_summary.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df_summary.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    print(f"\nProceso completado. Resumen guardado en:")
    print(f"- {csv_path}")
    print(f"- {parquet_path}")
    return df_summary


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    run_pipeline(limit=1 if test_mode else None)
