"""
Parser y Clasificador de Diarización
-------------------------------------
Procesa los payloads de Deepgram y convierte las utterances crudas en
diálogos estructurados con roles asignados (Agente, Cliente, Otro).

Funcionalidades:
- Clasificación heurística de hablantes por vocabulario especializado.
- Desdoblamiento de diálogos mono-speaker en llamadas humanas.
- Formateo de turnos de diálogo a Markdown con timestamps.
- Extracción de métricas base por llamada (palabras, tiempos, confianza).
"""

from pathlib import Path
from typing import Dict, Any, List


# ── Vocabulario de IVR / Buzón de voz / Grabadoras ──────────────────────────

IVR_KEYWORDS = [
    "marque 1", "marque 2", "marque 3", "marque 4",
    "para las opciones de entrega", "opciones de entrega",
    "su grabación llegó", "su grabacion llego", "tiempo límite", "tiempo limite",
    "buzón de voz", "buzon de voz", "deje su mensaje", "después del tono", "despues del tono",
    "la llamada será transferida", "la llamada sera transferida",
    "casilla de voz", "casilla de mensajes", "número que usted marcó", "numero que usted marco"
]

# ── Vocabulario de Agente IA ────────────────────────────────────────────────

AI_AGENT_KEYWORDS = [
    "embargos", "judicializaciones", "alivios financieros",
    "área de embargos", "area de embargos", "le llamo del area"
    "obligación", "obligacion", "obligaciones",
    "acuerdos de pago", "acuerdo de pago",
    "le hablo", "me comunico", "nos comunicamos",
    "tengo el gusto de hablar con", "me estoy comunicando",
    "dueños", "dueña", "dueño", "crediticio", "crediticia",
    "cartera", "cómo se encuentra", "como se encuentra",
    "podría confirmarme", "podria confirmarme",
    "hablo con", "me confirma"
]

# ── Vocabulario especializado de Agente Humano en cobranzas ─────────────────

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

# ── Patrones de respuestas típicas del cliente en cobranzas ─────────────────

CLIENT_TRIGGER_PATTERNS = [
    "con ella", "con él", "con el", "quién la necesita", "quien la necesita",
    "quién habla", "quien habla", "de parte de quién", "de parte de quien",
    "sí, con ella", "si, con ella", "sí con ella", "sí, con él",
    "no pude hacer el pago", "se me presentó una calamidad", "se me presento una calamidad",
    "no tengo dinero", "no tengo plata", "no cuento con", "estoy desempleado",
    "a mí me pagan", "a mi me pagan"
]

# Subconjuntos para la lógica de desdoblamiento de turnos
_CLIENT_SPLIT_CUES = [
    "con ella", "con él", "con el", "quién la necesita", "quien la necesita",
    "sí, con ella", "no pude hacer", "calamidad", "no tengo", "desempleado"
]
_AGENT_SPLIT_CUES = [
    "me estoy comunicando", "me comunico", "le habla", "obligación",
    "acuerdo de pago", "compromiso de pago", "compramos la cartera"
]


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
    speaker_texts: Dict[int, List[str]] = {s: [] for s in speakers}
    speaker_word_counts: Dict[int, int] = {s: 0 for s in speakers}
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
        scores: Dict[int, int] = {}
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
    roles: Dict[int, str] = {}
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
        if any(pat in txt_lower for pat in _CLIENT_SPLIT_CUES):
            current_spk = 1
        # Si coincide con patrón de agente
        elif any(pat in txt_lower for pat in _AGENT_SPLIT_CUES):
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
    source_type: str,
    transcripts_dir: Path
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
    confidences: List[float] = []

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
    md_path = transcripts_dir / f"{call_id}.md"
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
