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


# ── Extracción con LLM y Manejo de Caché ───────────────────────────────────────

def extract_features_for_call(
    client: httpx.Client,
    call_id: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    Extrae variables semánticas para una llamada individual.
    Si ya existe data/raw_features/{call_id}.json y no se usa force=True,
    carga desde caché y omite cualquier llamada a la API de Groq.
    """
    cache_file = RAW_FEATURES_DIR / f"{call_id}.json"

    if not force and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Validar con Pydantic para asegurar que la caché es consistente
            CallSemanticFeatures(**cached)
            return cached
        except Exception:
            # Si el archivo en caché estaba corrupto, se recalcula
            pass

    transcript_file = TRANSCRIPTS_DIR / f"{call_id}.md"
    if not transcript_file.exists():
        raise FileNotFoundError(f"No existe transcripción para {call_id} en {transcript_file}")

    transcript_content = transcript_file.read_text(encoding="utf-8")

    user_prompt = f"A continuación se presenta la transcripción de la llamada `{call_id}`:\n\n{transcript_content}\n\nAnaliza y entrega el JSON con las variables requeridas."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 400
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=60.0
            )

            if response.status_code == 200:
                res_json = response.json()
                content = res_json["choices"][0]["message"]["content"]
                parsed_data = json.loads(content)

                # Validar estrictamente con Pydantic
                validated = CallSemanticFeatures(**parsed_data)
                result_dict = validated.model_dump()

                # Guardar en caché individual de inmediato
                with open(cache_file, "w", encoding="utf-8") as f_out:
                    json.dump(result_dict, f_out, ensure_ascii=False, indent=2)

                return result_dict

            elif response.status_code == 429:
                wait_time = attempt * 5
                time.sleep(wait_time)
                last_err = f"Rate limit 429: {response.text}"
            else:
                last_err = f"HTTP {response.status_code}: {response.text}"
                time.sleep(2)

        except Exception as e:
            last_err = str(e)
            time.sleep(attempt * 2)

    raise RuntimeError(f"Error procesando {call_id} tras {max_retries} intentos: {last_err}")


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
# 'humano_43' se excluye deliberadamente por las siguientes razones técnicas y analíticas:
# 1. Valor atípico extremo (outlier): Su duración es de 20 min 33 s (1,233.2 s) con
#    más de 4,130 palabras (~11,500 tokens), superando en más de 5 desviaciones estándar
#    la duración promedio del dataset (~180 s). Incluirla distorsionaría las pruebas
#    paramétricas de duración y tiempos de habla.
# 2. Restricción técnica de tokens (ITPM): Su volumen excede el límite de tokens de entrada
#    por minuto (ITPM: 7,000 - 8,000) de los endpoints de LLM en el nivel estándar.
# 3. Preservación del rigor: El análisis mantiene 99 llamadas íntegras (49 humanas vs. 50 IA),
#    una muestra balanceada y representativa para contrastes no paramétricos y modelos logísticos.
EXCLUDED_CALLS = {"humano_43"}
EXCLUDED_CALLS_JUSTIFICATION = {
    "humano_43": (
        "Outlier extremo de duración (1,233.2 s / 20.5 min vs. media de ~180 s; >5 sigma) "
        "y longitud textual (~11,500 tokens), superando límites de ITPM de la API y "
        "con riesgo de sesgar desproporcionadamente las métricas acústico-temporales del grupo humano."
    )
}


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
