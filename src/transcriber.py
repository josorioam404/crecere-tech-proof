"""
Transcriptor y Diarizador de Audios con Deepgram nova-3
------------------------------------------------------
Procesa llamadas de audio (.wav) tanto humanas como de IA, aplicando
diarización de hablantes y extrayendo transcripciones estructuradas.

Cumple con la arquitectura de 3 capas:
1. data/raw_transcripts/{id}.json (Caché e idempotencia de Deepgram)
2. data/transcripts/{id}.md       (Diálogo formateado por turnos)
3. data/processed/transcripts_summary.csv / .parquet (Metadatos analíticos)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import httpx
from tqdm import tqdm
import pandas as pd


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

    # Carga desde .env si existe
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

    # Fallback a deepgram_secret.txt si existiese
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
    api_key: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Envía un archivo de audio a Deepgram API con el modelo nova-3 y diarización activada.
    Implementa reintentos con backoff exponencial.
    """
    audio_id = file_path.stem
    cache_path = RAW_DIR / f"{audio_id}.json"

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
                # Guardar en Capa 1
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

    raise RuntimeError(f"Fallo al transcribir {file_path.name}: {last_err}")


def format_seconds(seconds: float) -> str:
    """Formatea segundos a formato MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def determine_speaker_roles(utterances: List[Dict[str, Any]], source_type: str) -> Tuple[int, int]:
    """
    Identifica qué speaker id corresponde al Agente y cuál al Cliente.

    Reglas:
    - En llamadas de IA: La IA invariablemente dice frases institucionales como:
      'le llamo del area de embargos, judicializaciones y alivios financieros'.
      Identificamos al hablante que pronuncia estas palabras como el Agente IA.
    - En llamadas Humanas: No hay una frase específica. Se utiliza la heurística de apertura:
      el hablante que inicia o pregunta por el titular en los primeros turnos es clasificado
      como Agente.
    """
    if not utterances:
        return 0, 1

    speakers = list({u.get("speaker", 0) for u in utterances})
    if len(speakers) <= 1:
        return (speakers[0] if speakers else 0), -1

    # Regla específica para IA
    if source_type == "ia":
        ai_keywords = ["embargos", "judicializaciones", "alivios financieros", "área de embargos", "area de embargos"]
        for u in utterances:
            text = u.get("transcript", "").lower()
            if any(kw in text for kw in ai_keywords):
                agent_id = u.get("speaker", 0)
                customer_candidates = [s for s in speakers if s != agent_id]
                return agent_id, (customer_candidates[0] if customer_candidates else -1)

    # Heurística para llamadas humanas (o fallback en IA si no se detectó la frase)
    speaker_scores = {s: 0 for s in speakers}
    for idx, u in enumerate(utterances[:4]):
        text = u.get("transcript", "").lower()
        spk = u.get("speaker", 0)
        # Quien habla de primero tiene una probabilidad natural más alta de apertura
        if idx == 0:
            speaker_scores[spk] += 2
        # Patrones comunes de apertura de cobranza humana
        if "me comunico" in text or "hablo con" in text or "le llamo" in text:
            speaker_scores[spk] += 4

    agent_id = max(speaker_scores, key=speaker_scores.get)
    customer_candidates = [s for s in speakers if s != agent_id]
    customer_id = customer_candidates[0] if customer_candidates else -1

    return agent_id, customer_id


def process_and_save_dialogue(
    raw_data: Dict[str, Any],
    audio_id: str,
    source_type: str
) -> Dict[str, Any]:
    """
    Convierte el payload de Deepgram en un diálogo legible (Capa 2) y extrae
    métricas base para la tabla analítica (Capa 3).
    """
    results = raw_data.get("results", {})
    utterances = results.get("utterances", [])
    metadata = raw_data.get("metadata", {})
    duration = metadata.get("duration", 0.0)

    # Si no hay utterances pero hay canales
    if not utterances:
        channels = results.get("channels", [{}])
        alt = channels[0].get("alternatives", [{}])[0]
        full_transcript = alt.get("transcript", "")
        utterances = [{
            "speaker": 0,
            "start": 0.0,
            "end": duration,
            "transcript": full_transcript,
            "confidence": alt.get("confidence", 0.0)
        }]

    agent_id, customer_id = determine_speaker_roles(utterances, source_type)

    dialogue_lines = [
        f"# Transcripción de Llamada — {audio_id}",
        f"- **Origen:** {source_type.upper()}",
        f"- **Duración:** {format_seconds(duration)} ({duration:.1f} s)",
        f"- **Hablante Agente Identificado:** Hablante {agent_id}",
        f"- **Hablante Cliente Identificado:** Hablante {customer_id if customer_id != -1 else 'No detectado'}",
        "",
        "## Diálogo",
        ""
    ]

    total_words = 0
    agent_words = 0
    customer_words = 0
    agent_talk_time = 0.0
    customer_talk_time = 0.0
    confidences = []

    for u in utterances:
        spk = u.get("speaker", 0)
        start = u.get("start", 0.0)
        end = u.get("end", 0.0)
        text = u.get("transcript", "").strip()
        conf = u.get("confidence", 0.0)
        confidences.append(conf)

        u_duration = max(0.0, end - start)
        words_count = len(text.split())
        total_words += words_count

        if spk == agent_id:
            role_label = "Agente"
            agent_words += words_count
            agent_talk_time += u_duration
        elif spk == customer_id:
            role_label = "Cliente"
            customer_words += words_count
            customer_talk_time += u_duration
        else:
            role_label = f"Hablante {spk}"

        dialogue_lines.append(f"**[{format_seconds(start)} - {format_seconds(end)}] {role_label}:** {text}")

    # Guardar en Capa 2
    md_path = TRANSCRIPTS_DIR / f"{audio_id}.md"
    with open(md_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(dialogue_lines))

    unique_speakers = len({u.get("speaker", 0) for u in utterances})

    return {
        "audio_id": audio_id,
        "source": source_type,
        "duration_seconds": round(duration, 2),
        "total_words": total_words,
        "agent_words": agent_words,
        "customer_words": customer_words,
        "agent_talk_time_seconds": round(agent_talk_time, 2),
        "customer_talk_time_seconds": round(customer_talk_time, 2),
        "talk_ratio_agent": round(agent_talk_time / duration, 4) if duration > 0 else 0.0,
        "num_speakers": unique_speakers,
        "num_utterances": len(utterances),
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    }


def run_pipeline(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo para todos los audios de ambas carpetas.
    Si limit se especifica, procesa solo esa cantidad de cada categoría para test.
    """
    setup_directories()
    api_key = load_api_key()

    human_files = sorted(list(HUMAN_DIR.glob("*.wav")))
    ai_files = sorted(list(AI_DIR.glob("*.wav")))

    if limit:
        human_files = human_files[:limit]
        ai_files = ai_files[:limit]

    all_tasks = [(f, "humano") for f in human_files] + [(f, "ia") for f in ai_files]
    print(f"Total de audios a procesar: {len(all_tasks)} (Humanos: {len(human_files)}, IA: {len(ai_files)})")

    summary_records = []
    with httpx.Client() as client:
        for file_path, source in tqdm(all_tasks, desc="Transcribiendo con Deepgram nova-3"):
            audio_id = file_path.stem
            try:
                raw_json = transcribe_audio_file(client, file_path, api_key)
                meta = process_and_save_dialogue(raw_json, audio_id, source)
                summary_records.append(meta)
            except Exception as err:
                print(f"Error procesando {file_path.name}: {err}", file=sys.stderr)

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
    # Permite pasar argumento opcional para prueba (--test)
    test_mode = "--test" in sys.argv
    run_pipeline(limit=1 if test_mode else None)
