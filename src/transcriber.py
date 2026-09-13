"""
Transcriptor y Diarizador de Audios con Deepgram nova-3
------------------------------------------------------
Procesa llamadas de audio (.wav) tanto humanas como de IA, aplicando
diarización de hablantes y extrayendo transcripciones estructuradas.

Clasificación de roles avanzada:
- Agente: Basado en frases institucionales de IA ("área de embargos, judicializaciones...")
          y vocabulario especializado en humanas ("obligación", "acuerdos de pago", "cartera"...).
- Cliente: Interlocutor principal que responde a la gestión.
- Otro: Sistemas automáticos (IVR, buzón de voz, "marque 1...") o terceros que intervienen.

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

# Vocabulario de IVR / Buzón de voz / Grabadoras
IVR_KEYWORDS = [
    "marque 1", "marque 2", "marque 3", "marque 4",
    "para las opciones de entrega", "opciones de entrega",
    "su grabación llegó", "su grabacion llego", "tiempo límite", "tiempo limite",
    "buzón de voz", "buzon de voz", "deje su mensaje", "después del tono", "despues del tono",
    "la llamada será transferida", "la llamada sera transferida",
    "casilla de voz", "casilla de mensajes", "número que usted marcó", "numero que usted marco"
]

# Vocabulario de Agente IA
AI_AGENT_KEYWORDS = [
    "embargos", "judicializaciones", "alivios financieros",
    "área de embargos", "area de embargos", "le llamo del area"
]

# Vocabulario especializado de Agente Humano en cobranzas
HUMAN_AGENT_KEYWORDS = [
    "obligación", "obligacion", "obligaciones",
    "acuerdos de pago", "acuerdo de pago",
    "le hablo", "me comunico", "nos comunicamos",
    "tengo el gusto de hablar con", "me estoy comunicando",
    "dueños", "dueña", "dueño", "crediticio", "crediticia",
    "cartera", "cómo se encuentra", "como se encuentra",
    "podría confirmarme", "podria confirmarme",
    "hablo con", "me confirma"
]


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


def format_seconds(seconds: float) -> str:
    """Formatea segundos a formato MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def classify_speakers(
    utterances: List[Dict[str, Any]],
    source_type: str
) -> Dict[int, str]:
    """
    Clasifica a cada hablante detectado en:
    - 'Agente'
    - 'Cliente'
    - 'Otro' (Buzón, IVR, contestadora o tercero)
    """
    if not utterances:
        return {0: "Agente"}

    speakers = sorted(list({u.get("speaker", 0) for u in utterances}))

    # 1. Agrupar texto completo por hablante
    speaker_texts = {s: [] for s in speakers}
    speaker_word_counts = {s: 0 for s in speakers}
    for u in utterances:
        spk = u.get("speaker", 0)
        txt = u.get("transcript", "")
        speaker_texts[spk].append(txt)
        speaker_word_counts[spk] += len(txt.split())

    full_texts = {s: " ".join(speaker_texts[s]).lower() for s in speakers}

    # 2. Identificar hablantes tipo IVR / Buzón / Grabadora -> 'Otro'
    is_ivr = {}
    for s in speakers:
        text = full_texts[s]
        is_ivr[s] = any(kw in text for kw in IVR_KEYWORDS)

    # Si todos los hablantes son IVR, o el único hablante es IVR -> 'Otro'
    if all(is_ivr[s] for s in speakers):
        return {s: "Otro" for s in speakers}

    # 3. Identificar Agente
    agent_id = None

    if source_type == "ia":
        # En IA, buscar frase insignia
        for s in speakers:
            if not is_ivr[s] and any(kw in full_texts[s] for kw in AI_AGENT_KEYWORDS):
                agent_id = s
                break
        if agent_id is None:
            # Fallback en IA: el hablante no-IVR que más habla en los turnos iniciales
            candidates = [s for s in speakers if not is_ivr[s]]
            agent_id = candidates[0] if candidates else speakers[0]
    else:
        # En llamadas humanas: puntuar con términos específicos de agentes de cobranza
        scores = {}
        for s in speakers:
            if is_ivr[s]:
                scores[s] = -100
                continue
            text = full_texts[s]
            score = 0
            for kw in HUMAN_AGENT_KEYWORDS:
                score += text.count(kw) * 3

            # Bonificación por iniciativa de apertura (primeros turnos)
            for idx, u in enumerate(utterances[:3]):
                if u.get("speaker", 0) == s:
                    score += (3 - idx) * 2

            scores[s] = score

        best_spk = max(scores, key=scores.get)
        agent_id = best_spk if scores[best_spk] > -50 else speakers[0]

    # 4. Asignar roles finales
    roles = {}
    roles[agent_id] = "Agente"

    remaining_speakers = [s for s in speakers if s != agent_id]

    # Clasificar el resto
    primary_customer_id = None
    max_customer_words = -1

    for s in remaining_speakers:
        if is_ivr[s]:
            roles[s] = "Otro"
        else:
            # Candidato a cliente principal: quien tiene mayor interacción
            if speaker_word_counts[s] > max_customer_words:
                max_customer_words = speaker_word_counts[s]
                primary_customer_id = s

    if primary_customer_id is not None:
        roles[primary_customer_id] = "Cliente"

    # Cualquier otro hablante adicional no-IVR ni cliente principal -> 'Otro'
    for s in remaining_speakers:
        if s not in roles:
            roles[s] = "Otro"

    return roles


def separate_single_speaker_dialogue(
    utterances: List[Dict[str, Any]],
    source_type: str
) -> List[Dict[str, Any]]:
    """
    Si Deepgram agrupó toda la llamada bajo un solo hablante (speaker 0),
    pero la conversación contiene respuestas evidentes del cliente,
    desdobla los turnos asignando speaker 0 (Agente) y speaker 1 (Cliente).
    """
    if source_type != "humano" or not utterances:
        return utterances

    speakers = list({u.get("speaker", 0) for u in utterances})
    if len(speakers) > 1:
        return utterances

    # Frases típicas de respuestas del cliente en cobranzas
    CLIENT_TRIGGER_PATTERNS = [
        "con ella", "con él", "con el", "quién la necesita", "quien la necesita",
        "quién habla", "quien habla", "de parte de quién", "de parte de quien",
        "sí, con ella", "si, con ella", "sí con ella", "sí, con él",
        "no pude hacer el pago", "se me presentó una calamidad", "se me presento una calamidad",
        "no tengo dinero", "no tengo plata", "no cuento con", "estoy desempleado",
        "a mí me pagan", "a mi me pagan"
    ]

    has_client_cue = False
    for u in utterances:
        txt = u.get("transcript", "").lower()
        if any(pat in txt for pat in CLIENT_TRIGGER_PATTERNS):
            has_client_cue = True
            break

    if not has_client_cue:
        return utterances

    # Desdoblar turnos conversacionales
    new_utterances = []
    current_spk = 0  # El agente suele iniciar el saludo institucional

    for u in utterances:
        u_copy = dict(u)
        txt = u.get("transcript", "").strip()
        txt_lower = txt.lower()

        # Si coincide con patrón de cliente
        if any(pat in txt_lower for pat in ["con ella", "con él", "con el", "quién la necesita", "quien la necesita", "sí, con ella", "no pude hacer", "calamidad", "no tengo", "desempleado"]):
            current_spk = 1
        # Si coincide con patrón de agente
        elif any(pat in txt_lower for pat in ["me estoy comunicando", "me comunico", "le habla", "obligación", "acuerdo de pago", "compromiso de pago", "compramos la cartera"]):
            current_spk = 0

        u_copy["speaker"] = current_spk
        new_utterances.append(u_copy)

        # Transición conversacional natural tras una pregunta del agente
        if current_spk == 0 and (txt.endswith("?") or "¿" in txt):
            current_spk = 1
        elif current_spk == 1 and not (txt.endswith("?") or "¿" in txt):
            current_spk = 0

    return new_utterances


def process_and_save_dialogue(
    raw_data: Dict[str, Any],
    call_id: str,
    original_audio_file: str,
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

    # Si no hay utterances pero hay alternativas de canales
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

    # Desdoblar hablantes si Deepgram unificó erróneamente en speaker 0 en llamadas humanas
    utterances = separate_single_speaker_dialogue(utterances, source_type)
    speaker_roles = classify_speakers(utterances, source_type)

    # Construir encabezado
    dialogue_lines = [
        f"# Transcripción de Llamada — {call_id}",
        f"- **ID de Llamada:** `{call_id}`",
        f"- **Archivo Original:** `{original_audio_file}`",
        f"- **Origen:** {source_type.upper()}",
        f"- **Duración:** {format_seconds(duration)} ({duration:.1f} s)",
        f"- **Hablantes Detectados:**"
    ]
    for spk_id, role in speaker_roles.items():
        dialogue_lines.append(f"  - Hablante {spk_id}: **{role}**")

    dialogue_lines.extend(["", "## Diálogo", ""])

    total_words = 0
    words_by_role = {"Agente": 0, "Cliente": 0, "Otro": 0}
    time_by_role = {"Agente": 0.0, "Cliente": 0.0, "Otro": 0.0}
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

        role = speaker_roles.get(spk, "Otro")
        words_by_role[role] = words_by_role.get(role, 0) + words_count
        time_by_role[role] = time_by_role.get(role, 0.0) + u_duration

        dialogue_lines.append(f"**[{format_seconds(start)} - {format_seconds(end)}] {role}:** {text}")

    # Guardar en Capa 2 con el call_id estandarizado
    md_path = TRANSCRIPTS_DIR / f"{call_id}.md"
    with open(md_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(dialogue_lines))

    unique_speakers = len(speaker_roles)
    roles_list = sorted(list(set(speaker_roles.values())))

    return {
        "call_id": call_id,
        "original_audio_file": original_audio_file,
        "source": source_type,
        "duration_seconds": round(duration, 2),
        "total_words": total_words,
        "agent_words": words_by_role.get("Agente", 0),
        "customer_words": words_by_role.get("Cliente", 0),
        "other_words": words_by_role.get("Otro", 0),
        "agent_talk_time_seconds": round(time_by_role.get("Agente", 0.0), 2),
        "customer_talk_time_seconds": round(time_by_role.get("Cliente", 0.0), 2),
        "other_talk_time_seconds": round(time_by_role.get("Otro", 0.0), 2),
        "talk_ratio_agent": round(time_by_role.get("Agente", 0.0) / duration, 4) if duration > 0 else 0.0,
        "num_speakers": unique_speakers,
        "has_other_speaker": "Otro" in roles_list,
        "roles_detected": ", ".join(roles_list),
        "num_utterances": len(utterances),
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    }


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
                meta = process_and_save_dialogue(raw_json, call_id, file_path.name, source)
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
