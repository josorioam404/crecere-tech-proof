"""
Extractor Estructurado de Features con Groq (openai/gpt-oss-120b)
----------------------------------------------------------------
Analiza las transcripciones de llamadas en Markdown (Capa 2) mediante un LLM
de alta capacidad para extraer variables semánticas de contactabilidad,
conversión, objeciones, cumplimiento normativo y claridad del mensaje.

Características clave:
1. Idempotencia y caché granular: data/raw_features/{call_id}.json.
   Las llamadas en caché se cargan de disco en milisegundos sin consumir tokens.
2. Control por CLI: soporte para --range, --ids, --source, --limit, --force.
3. Validación estricta con Pydantic y reintentos con backoff exponencial.
4. Fusión con métricas acústico-temporales de Deepgram en analytical_dataset.csv / .parquet.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
import httpx
from tqdm import tqdm
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ValidationError


# ── Rutas del Proyecto ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_FEATURES_DIR = DATA_DIR / "raw_features"
SUMMARY_CSV = PROCESSED_DIR / "transcripts_summary.csv"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


# ── Esquema Pydantic para Validación Estricta ─────────────────────────────────

class CallSemanticFeatures(BaseModel):
    is_rpc: bool = Field(
        description="True si se valida contacto efectivo con el titular deudor. False si es tercero, buzón, IVR o número equivocado."
    )
    ptp_logrado: bool = Field(
        description="True si el cliente asume un compromiso formal de pago con fecha, monto o canal acordado. False en caso contrario."
    )
    num_objeciones: int = Field(
        ge=0,
        description="Número de objeciones o razones de no pago manifestadas por el cliente (falta de liquidez, desempleo, desconocimiento, etc.)."
    )
    num_objeciones_resueltas: int = Field(
        ge=0,
        description="Número de objeciones que el agente logró resolver o rebatir favorablemente llevando al cliente hacia una solución."
    )
    alternativas_ofrecidas: int = Field(
        ge=0,
        description="Número de opciones de alivio ofrecidas por el agente (pago a cuotas, abono mínimo, descuento de intereses, prórroga de fecha)."
    )
    tono_predominante_agente: Literal[
        "empatico_profesional",
        "neutro_formal",
        "rigido_robotico",
        "confrontativo"
    ] = Field(
        description="Tono predominante del agente durante la interacción."
    )
    peticiones_aclaracion_cliente: int = Field(
        ge=0,
        description="Número de veces que el cliente pide repetición o aclaración ('¿cómo?', '¿qué dijo?', 'no entendí', '¿me repite?')."
    )
    justificacion_breve: str = Field(
        description="Resumen conciso (1-2 frases) que justifica las clasificaciones asignadas."
    )


# ── Prompt del Sistema para el LLM ────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un auditor senior y científico de datos especializado en análisis de calidad y operaciones de cobranza bancaria y BPO.
Tu misión es analizar la transcripción de una llamada de cobranza y extraer variables cuantitativas y categóricas estrictas sobre el diálogo.

Debes responder ÚNICAMENTE con un objeto JSON válido que cumpla rigurosamente con esta especificación:

{
  "is_rpc": <bool>,
  "ptp_logrado": <bool>,
  "num_objeciones": <int >= 0>,
  "num_objeciones_resueltas": <int >= 0>,
  "alternativas_ofrecidas": <int >= 0>,
  "tono_predominante_agente": <"empatico_profesional" | "neutro_formal" | "rigido_robotico" | "confrontativo">,
  "peticiones_aclaracion_cliente": <int >= 0>,
  "justificacion_breve": <string>
}

CRITERIOS DE CLASIFICACIÓN OPERATIVA:
1. is_rpc (Right Party Contact):
   - True: Se confirma inequívocamente que la persona que habla es el titular deudor.
   - False: Responde una contestadora/buzón/IVR, un familiar o tercero que no es el titular, o es número equivocado.

2. ptp_logrado (Promise to Pay):
   - True: El cliente acepta formalmente pagar en una fecha establecida, bajo un plan de cuotas o un abono con canal definido (ej. Efecty, Bancolombia, etc.).
   - False: El cliente no se compromete, rechaza pagar, cuelga o la llamada no tuvo contacto con el titular.

3. num_objeciones:
   - Conteo de objeciones expresadas por el cliente (ej. "no tengo plata", "estoy desempleado", "esa deuda ya la pagué", "cobran mucho interés", "se me presentó una calamidad").
   - Si no hubo objeciones (o no hubo contacto con titular), colocar 0.

4. num_objeciones_resueltas:
   - Cuántas de esas objeciones fueron manejadas eficazmente por el agente logrando que el cliente acceda a continuar o cerrar un acuerdo.
   - Siempre debe ser <= num_objeciones.

5. alternativas_ofrecidas:
   - Conteo de herramientas de negociación brindadas por el agente: ofrecer pago en cuotas, descuento de intereses/honorarios, opción de abono parcial, aplazamiento a una fecha más conveniente.
   - Si el agente solo exige el pago total sin dar ninguna flexibilidad, colocar 0.

6. tono_predominante_agente:
   - "empatico_profesional": Escucha activa, tono cálido, comprensivo ante calamidades pero enfocado en buscar soluciones.
   - "neutro_formal": Protocolario, respetuoso, directo, estándar institucional sin mayor carga emocional.
   - "rigido_robotico": Monótono, repite guion sin adaptarse a lo que dice el cliente, poca flexibilidad conversacional.
   - "confrontativo": Intimidatorio, impaciente, agresivo o tajante.

7. peticiones_aclaracion_cliente:
   - Veces que el cliente dice "¿cómo?", "¿qué?", "no le escuché bien", "¿me repite?", "no entendí nada", expresando confusión con lo que el agente acaba de explicar.

Sé objetivo, neutral y riguroso. No infieras datos que no estén en la transcripción. Responde estrictamente el JSON sin texto introductorio ni bloques markdown envolventes.
"""


# ── Funciones de Utilidad y Autenticación ──────────────────────────────────────

def load_api_key() -> str:
    """
    Carga la API key de Groq desde el entorno, .env o grok_secret.txt.
    """
    key = os.environ.get("GROQ_API_KEY")
    if key and key.strip():
        return key.strip()

    env_file = BASE_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        os.environ["GROQ_API_KEY"] = val
                        return val

    for secret_name in ["grok_secret.txt", "groq_secret.txt"]:
        secret_file = BASE_DIR / secret_name
        if secret_file.exists():
            val = secret_file.read_text(encoding="utf-8").strip()
            if val:
                os.environ["GROQ_API_KEY"] = val
                return val

    raise ValueError(
        "No se encontró GROQ_API_KEY en variables de entorno, archivo .env ni grok_secret.txt"
    )


def setup_directories() -> None:
    """Crea los directorios requeridos."""
    RAW_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)



# ── Chunking de Transcripts Largos ────────────────────────────────────────────

CHUNK_SIZE_CHARS = 3200   # Balance óptimo entre contexto y límite de TPM de Groq (8,000 TPM)
OVERLAP_TURNS = 2         # Turnos solapados entre chunks para preservar contexto

def split_transcript_into_chunks(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = OVERLAP_TURNS) -> List[str]:
    """
    Divide un transcript Markdown en chunks por turnos de diálogo.
    Cada turno comienza con '**['. Los chunks se solapan `overlap` turnos
    para preservar el contexto entre fragmentos.
    Retorna una lista de strings; si el texto cabe en un solo chunk, retorna [text].
    """
    if len(text) <= chunk_size:
        return [text]

    # Extraer encabezado (hasta "## Diálogo")
    header_end = text.find("## Diálogo")
    if header_end == -1:
        header = ""
        body = text
    else:
        header = text[:header_end + len("## Diálogo") + 1]
        body = text[header_end + len("## Diálogo") + 1:]

    # Dividir el body en turnos individuales
    turns = []
    current = []
    for line in body.splitlines(keepends=True):
        if line.startswith("**[") and current:
            turns.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        turns.append("".join(current))

    if not turns:
        return [text]

    chunks = []
    i = 0
    while i < len(turns):
        # Acumular turnos hasta llenar el chunk_size
        chunk_turns = []
        size = len(header)
        while i < len(turns) and size + len(turns[i]) <= chunk_size:
            chunk_turns.append(turns[i])
            size += len(turns[i])
            i += 1
        # Si no avanzó ni un turno (turno individual demasiado largo), incluirlo de todos modos
        if not chunk_turns and i < len(turns):
            chunk_turns.append(turns[i])
            i += 1

        chunks.append(header + "\n" + "".join(chunk_turns))

        # Retroceder `overlap` turnos para el solapamiento
        if i < len(turns):
            i = max(i - overlap, i - len(chunk_turns) + 1, i - overlap)

    return chunks if chunks else [text]


def aggregate_chunk_features(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combina los resultados de múltiples chunks en un único set de features.

    Reglas semánticas de agregación:
    - is_rpc, ptp_logrado     → OR  (basta detectarlo en cualquier chunk)
    - num_objeciones          → SUM (cada objeción cuenta)
    - num_objeciones_resueltas→ SUM, capped ≤ num_objeciones final
    - alternativas_ofrecidas  → SUM (suma de todas las opciones ofrecidas)
    - tono_predominante_agente→ MODE (tono más frecuente entre chunks)
    - peticiones_aclaracion   → SUM (acumulado en toda la llamada)
    - justificacion_breve     → primer chunk + " [...] " + último chunk
    """
    if not chunk_results:
        raise ValueError("chunk_results está vacío; no hay features que agregar.")

    if len(chunk_results) == 1:
        return chunk_results[0]

    is_rpc = any(c.get("is_rpc", False) for c in chunk_results)
    ptp_logrado = any(c.get("ptp_logrado", False) for c in chunk_results)

    num_obj = sum(c.get("num_objeciones", 0) for c in chunk_results)
    num_res = sum(c.get("num_objeciones_resueltas", 0) for c in chunk_results)
    num_res = min(num_res, num_obj)  # invariante del schema

    alt = sum(c.get("alternativas_ofrecidas", 0) for c in chunk_results)
    pet = sum(c.get("peticiones_aclaracion_cliente", 0) for c in chunk_results)

    tonos = [c.get("tono_predominante_agente", "neutro_formal") for c in chunk_results]
    tono = max(set(tonos), key=tonos.count)  # moda

    just_first = chunk_results[0].get("justificacion_breve", "")
    just_last = chunk_results[-1].get("justificacion_breve", "")
    if just_first == just_last:
        justificacion = just_first
    else:
        justificacion = f"{just_first} [...] {just_last}"

    return {
        "is_rpc": is_rpc,
        "ptp_logrado": ptp_logrado,
        "num_objeciones": num_obj,
        "num_objeciones_resueltas": num_res,
        "alternativas_ofrecidas": alt,
        "tono_predominante_agente": tono,
        "peticiones_aclaracion_cliente": pet,
        "justificacion_breve": justificacion,
    }


# ── Extracción con LLM y Manejo de Caché ───────────────────────────────────────

def extract_features_for_call(
    client: httpx.Client,
    call_id: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    max_retries: int = 12
) -> Dict[str, Any]:
    """
    Extrae variables semánticas para una llamada individual.

    Para transcripts largos (> CHUNK_SIZE_CHARS), divide automáticamente
    en chunks por turnos de diálogo y agrega los resultados con
    aggregate_chunk_features(). La caché final almacena el resultado
    ya agregado, idéntico al formato estándar.

    Si ya existe data/raw_features/{call_id}.json y no se usa force=True,
    carga desde caché y omite cualquier llamada a la API de Groq.
    """
    cache_file = RAW_FEATURES_DIR / f"{call_id}.json"

    if not force and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            CallSemanticFeatures(**cached)
            return cached
        except Exception:
            pass

    transcript_file = TRANSCRIPTS_DIR / f"{call_id}.md"
    if not transcript_file.exists():
        raise FileNotFoundError(f"No existe transcripción para {call_id} en {transcript_file}")

    transcript_content = transcript_file.read_text(encoding="utf-8")

    # ── Chunking automático para transcripts largos ───────────────────────────
    chunks = split_transcript_into_chunks(transcript_content)
    if len(chunks) > 1:
        print(f"  [chunking] {call_id}: {len(transcript_content):,} chars → {len(chunks)} chunks.")

    req_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    def _call_llm(chunk_text: str, chunk_label: str) -> Dict[str, Any]:
        """Llama al LLM con reintentos para un chunk dado."""
        user_prompt = (
            f"A continuación se presenta la transcripción de la llamada `{chunk_label}`:\n\n"
            f"{chunk_text}\n\nAnaliza y entrega el JSON con las variables requeridas."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 1000
        }
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.post(
                    GROQ_API_URL,
                    headers=req_headers,
                    json=payload,
                    timeout=60.0
                )
                if response.status_code == 200:
                    res_json = response.json()
                    content_str = res_json["choices"][0]["message"]["content"]
                    parsed_data = json.loads(content_str)
                    validated = CallSemanticFeatures(**parsed_data)
                    return validated.model_dump()
                elif response.status_code == 429:
                    wait_time = 35.0
                    try:
                        err_data = response.json()
                        err_msg = err_data.get("error", {}).get("message", "")
                        import re
                        m = re.search(r"try again in ([\d\.]+)s", err_msg)
                        if m:
                            wait_time = max(35.0, float(m.group(1)) + 15.0)
                    except Exception:
                        wait_time = max(35.0, attempt * 10.0)
                    print(f"\n  [Rate Limit 429 en {chunk_label}] Esperando {wait_time:.1f}s para vaciar ventana TPM...")
                    time.sleep(wait_time)
                    last_err = f"Rate limit 429: {response.text}"
                else:
                    last_err = f"HTTP {response.status_code}: {response.text}"
                    time.sleep(2)
            except Exception as e:
                last_err = str(e)
                time.sleep(attempt * 2)
        raise RuntimeError(f"Error en chunk '{chunk_label}' tras {max_retries} intentos: {last_err}")

    # Procesar todos los chunks con persistencia incremental
    chunks_cache_file = RAW_FEATURES_DIR / f"{call_id}_chunks_partial.json"
    chunk_results = []
    if chunks_cache_file.exists():
        try:
            with open(chunks_cache_file, "r", encoding="utf-8") as f_chk:
                chunk_results = json.load(f_chk)
            print(f"  [resumiendo] Se recuperaron {len(chunk_results)}/{len(chunks)} chunks de caché parcial.")
        except Exception:
            chunk_results = []

    for idx in range(len(chunk_results), len(chunks)):
        chunk_text = chunks[idx]
        chunk_label = f"{call_id}_chunk{idx + 1}of{len(chunks)}" if len(chunks) > 1 else call_id
        chunk_result = _call_llm(chunk_text, chunk_label)
        chunk_results.append(chunk_result)
        with open(chunks_cache_file, "w", encoding="utf-8") as f_chk:
            json.dump(chunk_results, f_chk, ensure_ascii=False, indent=2)
        if len(chunks) > 1 and idx < len(chunks) - 1:
            time.sleep(25.0)  # Pausa preventiva para que los tokens acumulados en Groq caigan a 0

    # Agregar resultados de todos los chunks y guardar en caché definitiva
    result_dict = aggregate_chunk_features(chunk_results)
    with open(cache_file, "w", encoding="utf-8") as f_out:
        json.dump(result_dict, f_out, ensure_ascii=False, indent=2)

    # Limpiar caché parcial temporal
    if chunks_cache_file.exists():
        try:
            chunks_cache_file.unlink()
        except Exception:
            pass

    return result_dict


# ── Filtrado Granular de Tareas ────────────────────────────────────────────────

def resolve_target_calls(
    all_call_ids: List[str],
    range_arg: Optional[str] = None,
    ids_arg: Optional[str] = None,
    source_arg: Optional[str] = None,
    limit: Optional[int] = None
) -> List[str]:
    """
    Filtra la lista de call_ids según los criterios CLI especificados.
    """
    selected = list(all_call_ids)

    # 1. Filtro por origen (humano vs ia)
    if source_arg:
        s_lower = source_arg.lower().strip()
        if s_lower in ["humano", "humanos"]:
            selected = [c for c in selected if c.startswith("humano_")]
        elif s_lower in ["ia", "ai"]:
            selected = [c for c in selected if c.startswith("ia_")]

    # 2. Filtro por IDs específicos
    if ids_arg:
        explicit_ids = {x.strip() for x in ids_arg.split(",") if x.strip()}
        selected = [c for c in selected if c in explicit_ids]

    # 3. Filtro por rango (ej: humano_01:humano_20 o 1:20)
    if range_arg and ":" in range_arg:
        start_str, end_str = range_arg.split(":", 1)
        start_str, end_str = start_str.strip(), end_str.strip()

        # Si es rango numérico (ej. 1:20)
        if start_str.isdigit() and end_str.isdigit():
            start_idx = max(1, int(start_str)) - 1
            end_idx = int(end_str)
            selected = selected[start_idx:end_idx]
        else:
            # Si es por nombre de ID
            try:
                idx_start = selected.index(start_str)
                idx_end = selected.index(end_str) + 1
                selected = selected[idx_start:idx_end]
            except ValueError:
                # Si no están en la lista seleccionada actual, buscar en la lista completa
                if start_str in all_call_ids and end_str in all_call_ids:
                    idx_start = all_call_ids.index(start_str)
                    idx_end = all_call_ids.index(end_str) + 1
                    selected = [c for c in all_call_ids[idx_start:idx_end] if c in selected]

    # 4. Límite final
    if limit and limit > 0:
        selected = selected[:limit]

    return selected


# ── Exclusión Metodológica de Llamadas Atípicas ──────────────────────────────
# 'humano_43' fue originalmente excluida por restricción técnica de tokens (ITPM ~7,000-8,000
# de Groq) dado su volumen de ~11,500 tokens. Esta restricción fue RESUELTA mediante chunking
# automático del transcript con agregación semántica de resultados (ver extract_features_for_call).
# Para la distorsión acústica (duración 1,233.2 s / >5 sigma), se aplica Winsorización en
# statistical_analysis.py sobre las pruebas de H5 (talk_ratio / duración).
# La muestra queda balanceada: n=100 (50 humanos vs. 50 IA).
EXCLUDED_CALLS: set = set()
EXCLUDED_CALLS_JUSTIFICATION: dict = {}


# ── Fusión con Datos Acústicos y Ensamble Final ────────────────────────────────

def assemble_analytical_dataset() -> pd.DataFrame:
    """
    Combina todas las features semánticas en data/raw_features/ con las
    métricas acústicas de transcripts_summary.csv para generar
    data/processed/analytical_dataset.csv y .parquet.
    Omite llamadas excluidas (humano_43).
    """
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"No se encontró el archivo base {SUMMARY_CSV}")

    df_summary = pd.read_csv(SUMMARY_CSV)
    df_summary = df_summary[~df_summary["call_id"].isin(EXCLUDED_CALLS)].copy()

    feature_records = []
    for _, row in df_summary.iterrows():
        call_id = row["call_id"]
        feature_file = RAW_FEATURES_DIR / f"{call_id}.json"

        record: Dict[str, Any] = {"call_id": call_id}
        if feature_file.exists():
            try:
                with open(feature_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                record.update(data)
            except Exception:
                pass

        feature_records.append(record)

    df_features = pd.DataFrame(feature_records)
    df_merged = pd.merge(df_summary, df_features, on="call_id", how="left")

    # Cálculos derivados acordados
    # 1. Claridad del mensaje = 1 / (1 + peticiones_aclaracion_cliente)
    if "peticiones_aclaracion_cliente" in df_merged.columns:
        df_merged["claridad_mensaje"] = df_merged["peticiones_aclaracion_cliente"].apply(
            lambda x: round(1.0 / (1.0 + float(x)), 4) if pd.notna(x) else np.nan
        )

    # 2. Tasa de resolución de objeciones = resueltas / total objeciones
    if "num_objeciones" in df_merged.columns and "num_objeciones_resueltas" in df_merged.columns:
        def calc_res_rate(row):
            num_obj = row["num_objeciones"]
            res_obj = row["num_objeciones_resueltas"]
            if pd.isna(num_obj) or num_obj == 0:
                return np.nan
            return round(min(1.0, float(res_obj) / float(num_obj)), 4)

        df_merged["tasa_resolucion_objeciones"] = df_merged.apply(calc_res_rate, axis=1)

    # 3. Alias estandarizados en español para métricas temporales
    df_merged["duracion_total_segundos"] = df_merged["duration_seconds"]
    df_merged["talk_ratio_agente"] = df_merged["talk_ratio_agent"]

    # Exportar resultados en Parquet y CSV
    out_csv = PROCESSED_DIR / "analytical_dataset.csv"
    out_parquet = PROCESSED_DIR / "analytical_dataset.parquet"

    df_merged.to_csv(out_csv, index=False, encoding="utf-8")
    try:
        df_merged.to_parquet(out_parquet, index=False)
    except Exception as err:
        print(f"Advertencia al guardar parquet: {err}", file=sys.stderr)

    return df_merged


# ── Pipeline Principal de Ejecución ───────────────────────────────────────────

def run_extraction_pipeline(
    range_arg: Optional[str] = None,
    ids_arg: Optional[str] = None,
    source_arg: Optional[str] = None,
    limit: Optional[int] = None,
    force: bool = False,
    model: str = DEFAULT_MODEL
) -> None:
    """
    Ejecuta el pipeline de extracción granular con auditoría previa y barra de progreso.
    """
    setup_directories()
    api_key = load_api_key()

    # Obtener todas las llamadas registradas en Capa 2
    transcript_files = sorted(list(TRANSCRIPTS_DIR.glob("*.md")))
    all_call_ids = [f.stem for f in transcript_files if f.stem not in EXCLUDED_CALLS]

    # Ordenar asegurando humanos primero, luego IA
    humans = [c for c in all_call_ids if c.startswith("humano_")]
    ais = [c for c in all_call_ids if c.startswith("ia_")]
    all_call_ids = sorted(humans) + sorted(ais)

    target_calls = resolve_target_calls(all_call_ids, range_arg, ids_arg, source_arg, limit)

    if not target_calls:
        print("No se encontraron llamadas que coincidan con los filtros indicados.")
        return

    # Auditoría de estado antes de hacer llamadas a la API
    cached_calls = []
    pending_calls = []
    for cid in target_calls:
        cache_path = RAW_FEATURES_DIR / f"{cid}.json"
        if not force and cache_path.exists():
            cached_calls.append(cid)
        else:
            pending_calls.append(cid)

    print("\n" + "=" * 60)
    print(" Auditoría de Extracción de Variables — Groq (openai/gpt-oss-120b)")
    print("=" * 60)
    print(f"• Total llamadas seleccionadas : {len(target_calls)}")
    print(f"• Ya en caché (0 tokens Groq) : {len(cached_calls)}")
    print(f"• Pendientes por consultar     : {len(pending_calls)}")
    print(f"• Modelo en uso               : {model}")
    print(f"• Sobrescritura forzada (--force): {force}")
    print("=" * 60 + "\n")

    if pending_calls:
        with httpx.Client() as client:
            for cid in tqdm(pending_calls, desc="Consultando Groq API"):
                try:
                    extract_features_for_call(
                        client=client,
                        call_id=cid,
                        api_key=api_key,
                        model=model,
                        force=force
                    )
                    # Pequeña pausa de cortesía para no saturar el rate-limit de Groq
                    time.sleep(1.5)
                except Exception as err:
                    print(f"\n[ERROR] Fallo en {cid}: {err}", file=sys.stderr)
    else:
        print("Todas las llamadas seleccionadas ya se encuentran procesadas en caché local.")

    # Generar dataset analítico final fusionado
    df_final = assemble_analytical_dataset()
    print("\nDataset analítico actualizado con éxito:")
    print(f"- Filas totales: {len(df_final)}")
    print(f"- Archivo CSV:     {PROCESSED_DIR / 'analytical_dataset.csv'}")
    print(f"- Archivo Parquet: {PROCESSED_DIR / 'analytical_dataset.parquet'}")


# ── Entrypoint CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extractor semántico estructurado de llamadas de cobranza con Groq."
    )
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Rango de llamadas a procesar (ej. 'humano_01:humano_25' o '1:20')."
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Lista de IDs separados por coma (ej. 'humano_01,ia_05,ia_10')."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["humano", "ia"],
        help="Filtrar llamadas únicamente por origen ('humano' o 'ia')."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite máximo de llamadas a procesar."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Modo prueba rápida: procesa 1 llamada humana y 1 llamada de IA."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza el reprocesamiento incluso si el archivo ya existe en caché."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Modelo de Groq a utilizar (por defecto: {DEFAULT_MODEL})."
    )

    args = parser.parse_args()

    if args.test:
        run_extraction_pipeline(ids_arg="humano_01,ia_01", force=args.force, model=args.model)
    else:
        run_extraction_pipeline(
            range_arg=args.range,
            ids_arg=args.ids,
            source_arg=args.source,
            limit=args.limit,
            force=args.force,
            model=args.model
        )
