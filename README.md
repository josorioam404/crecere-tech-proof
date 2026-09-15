# Creceré AI — Technical Challenge: Human vs. AI Voice Agents Analysis

Este repositorio contiene la solución completa a la prueba técnica para el rol de **Data Scientist / Data Analyst Junior** en Creceré AI. El proyecto evalúa cuantitativa y econométricamente una muestra de **100 llamadas de cobranza (50 humanas y 50 de IA)** para determinar si existen diferencias estadísticamente sustentables en su desempeño y aislar los determinantes causales de la promesa de pago (PTP).

---

## 1. Preguntas, Hipótesis y Decisiones Analíticas

### ¿Qué queremos entender?
1. **Efectividad y Conversión:** ¿Es la IA inherentemente menos efectiva para lograr compromisos de pago (PTP), o existen variables operativas y de negociación que explican la brecha?
2. **Contactabilidad y Fricción:** ¿Dónde se originan las pérdidas de llamadas (buzones/IVRs vs. rechazo explícito del titular)?
3. **Conducta Conversacional:** ¿En qué aspectos del diálogo superan los humanos a la IA y qué ventajas de consistencia presenta el bot?

### Hipótesis Declaradas y Contrastadas
* **$H_1$ (Alternativas de Pago):** Los humanos ofrecen una mayor variedad y cantidad de alternativas de pago que la IA.
* **$H_2$ (Claridad del Mensaje):** La IA genera menor necesidad de aclaración/repetición en los clientes debido a su dicción sintética y estandarizada.
* **$H_3$ (Resolución de Objeciones):** Los humanos tienen una tasa de resolución de objeciones significativamente mayor que la IA.
* **$H_4$ (Variabilidad Tonal):** Los humanos presentan mayor diversidad tonal (empatía y modulación) medida por entropía de Shannon $H(X)$.
* **$H_5$ (Monopolización de la Llamada):** La IA monopoliza la conversación con un mayor ratio de tiempo de habla que los humanos ($>75\%$).

### Decisiones Metodológicas Clave
* **Corte por Contacto Efectivo (RPC):** La contactabilidad bruta tiene un sesgo de discado previo ($p = 0.0134$). Para evitar sesgo de selección en la evaluación de la habilidad conversacional, el análisis de negociación y modelado causal se evalúa sobre las **$n = 61$ llamadas con contacto efectivo verificado** ($n_{\text{humano}} = 37$, $n_{\text{ia}} = 24$).
* **Modelado Econométrico Multivariado:** Se formula una regresión logística $\text{logit}(P(\text{PTP}=1))$ controlando simultáneamente por condición de IA, oferta de alternativas, claridad, monopolización y objeciones, aislando el efecto propio de cada factor.

---

## 2. Construcción de Datos y Variables

El dataset analítico (`data/processed/analytical_dataset.parquet` y `.csv`) integra variables en dos dimensiones:

### A. Variables Acústico-Temporales (Deepgram nova-3)
* `duration_seconds`: Duración total del audio.
* `talk_ratio_agente`: Proporción del tiempo de habla correspondiente al agente.
* `agent_words` / `customer_words`: Conteo de palabras por rol mediante diarización.
* `num_utterances`: Número de turnos de diálogo.
* `mean_confidence`: Calidad media de confianza acústica del ASR.

### B. Variables Semánticas Estructuradas (Groq LLM + Validación Pydantic)
* `is_rpc` (Right Party Contact): Contacto efectivo verificado con el titular deudor.
* `ptp_logrado` (Promise to Pay): Compromiso formal de pago con fecha, monto o canal.
* `num_objeciones`: Conteo de causas de no pago manifestadas por el deudor.
* `num_objeciones_resueltas`: Objeciones rebatidas o acordadas favorablemente.
* `tasa_resolucion_objeciones`: Ratio $\frac{\text{resueltas}}{\text{total objeciones}}$.
* `alternativas_ofrecidas`: Conteo de herramientas de alivio propuestas (cuotas, descuentos, prórrogas).
* `tono_predominante_agente`: Clasificación categórica (`empatico_profesional`, `neutro_formal`, `rigido_robotico`, `confrontativo`).
* `peticiones_aclaracion_cliente`: Frecuencia de repeticiones pedidas por el deudor.
* `claridad_mensaje`: Métrica calculada como $\frac{1}{1 + \text{peticiones\_aclaracion}}$.

---

## 3. Comparación Humanos vs. IA: Resultados Estadísticos

### Embudo Operativo Global ($n = 100$)
| Etapa del Embudo | Humanos ($n=50$) | IA ($n=50$) | Brecha | Test Estadístico / Significancia |
| :--- | :---: | :---: | :---: | :--- |
| **Contacto Efectivo (RPC)** | **74.0%** (37) | **48.0%** (24) | +26.0 p.p. | $\chi^2 = 6.05$, Fisher $p = 0.0134$* ($\text{OR} = 3.08$) |
| **Negociación Iniciada \| RPC** | 100.0% (37/37) | 100.0% (24/24) | 0.0 p.p. | No hay rechazo al iniciar el diálogo |
| **Oferta de Alternativas \| RPC** | 73.0% (27/37) | 100.0% (24/24) | -27.0 p.p. | IA ofrece siempre cuotas fijas (IQR = 0.0 vs. 3.0) |
| **Compromiso de Pago \| RPC** | **62.2%** (23) | **50.0%** (12) | +12.2 p.p. | Fisher $p = 0.457$ (No significativo sin control) |

### Matriz de Validación de Hipótesis ($n_{\text{RPC}} = 61$)
| Hipótesis | Variable | Humanos ($n=37$) | IA ($n=24$) | Contraste | Valor $p$ | Tamaño del Efecto | Veredicto |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- | :---: |
| **$H_1$: Alternativas** | `alternativas_ofrecidas` | Mediana: 2.0 (IQR 3.0) | Mediana: 2.0 (IQR 0.0) | Mann-Whitney $U = 387.0$ | 0.8125 | Cliff's $\delta = -0.128$ (Despreciable) | **No confirmada** |
| **$H_2$: Claridad** | `claridad_mensaje` | 0 rep: 56.8% | 0 rep: 33.3% | Fisher exacto | 0.9806 | Cliff's $\delta = -0.181$ (Pequeño) | **No confirmada** |
| **$H_3$: Objeciones** | `tasa_resolucion` | **67.3%** | **41.2%** | Mann-Whitney $U = 316.5$ | **0.0393\*** | Cliff's $\delta = +0.281$ (Pequeño) | **CONFIRMADA** |
| **$H_4$: Tono** | `tono_predominante` | $H(X) = 1.25$ bits | $H(X) = 1.20$ bits | $\chi^2 = 0.40$ ($df=2$) | 0.8187 | $\Delta H = +0.05$ bits | **No confirmada** |
| **$H_5$: Monopolización**| `talk_ratio_agente` | Media: 49.5% | Media: 46.2% | Welch's $t = -1.06$ | 0.8520 | Hedges' $g = -0.279$ (Pequeño) | **No confirmada** |

### Modelo Econométrico Multivariado ($\text{logit}(P(\text{PTP}=1))$)
$$\text{logit}(P(\text{PTP}=1)) = \beta_0 + \beta_1 \text{is\_ia} + \beta_2 \text{alternativas} + \beta_3 \text{claridad} + \beta_4 \text{talk\_ratio} + \beta_5 \text{num\_objeciones}$$

* **Efecto Alternativas ($\beta_2 = +0.7338$):** **$\text{Odds Ratio} = 2.083$** ($p = 0.0422$*, IC 95%: $[1.026, 4.229]$). Cada alternativa ofrecida **duplica las probabilidades relativas de pago**.
* **Efecto Resistencia ($\beta_5 = -0.4227$):** **$\text{Odds Ratio} = 0.655$** ($p = 0.0293$*). Cada objeción reduce las odds de acuerdo en un $34.5\%$.
* **Efecto Propio de la IA ($\beta_1 = -0.8807$):** **No significativo** ($p = 0.1453$, IC 95%: $[0.127, 1.356]$). Controlando por alternativas y objeciones, la IA no es estadísticamente inferior al humano.

---

## 4. Hallazgos Clave Accionables

1. **La IA duplica sus cierres por cada alternativa de pago desplegada:** Cada opción de alivio adicional multiplica las probabilidades de compromiso por **$2.08\times$ ($\text{OR} = 2.08$, $p = 0.042$)**.
2. **El cliente no rechaza la IA por ser máquina:** En el $100\%$ de los contactos efectivos la IA entabló negociación formal; en el modelo multivariado el factor IA carece de significancia estadística adversa ($p = 0.145$).
3. **Pérdida crítica de llamadas en el discado ($52.0\%$):** La IA solo contacta al $48.0\%$ de deudores frente al $74.0\%$ humano, perdiendo llamadas en contestadoras e IVRs por ausencia de detección acústica previa.
4. **Rigidez de negociación en la IA:** El bot actual presenta un rango intercuartílico $\text{IQR} = 0.0$ (siempre ofrece exactamente 2 cuotas fijas), mientras los humanos adaptan de 0 a 5 alternativas ($\text{IQR} = 3.0$) resolviendo el $67.3\%$ de objeciones vs. $41.2\%$ de la IA ($p = 0.039$).
5. **Reparto conversacional balanceado:** El bot habla el $46.2\%$ del tiempo frente al $49.5\%$ de los humanos, descartando monopolización indebida de la llamada.

---

## 5. Estructura y Pipeline de Ejecución

```text
├── src/
│   ├── transcriber.py           # Ingesta y transcripción con Deepgram nova-3 y diarización
│   ├── diarization_parser.py    # Parseo de turnos, vocabulario especializado y roles (Agente/Cliente/Otro)
│   ├── feature_extractor.py     # Extractor semántico estructurado con LLM (Groq) + Pydantic + Chunking
│   ├── statistical_analysis.py  # Tests no paramétricos (Mann-Whitney, Fisher), econometría (Logit)
│   └── generate_report.py       # Compilador autónomo del reporte ejecutivo en HTML (máx 2 págs)
├── reporte_ejecutivo.html       # Reporte ejecutivo final (2 páginas, imprimible A4)
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Thought process, metodología y resultados
```

### Reproducción Paso a Paso

1. **Instalar entorno:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configurar variables:**
   Configurar `DEEPGRAM_API_KEY` y `GROQ_API_KEY` en `.env` (siguiendo `.env.example`).
3. **Ejecutar el pipeline analítico:**
   ```bash
   # 1. Transcripción y diarización acústica
   python src/transcriber.py

   # 2. Extracción semántica estructurada (idempotente con caché local)
   python src/feature_extractor.py

   # 3. Modelado estadístico y econométrico
   python src/statistical_analysis.py

   # 4. Generación del reporte ejecutivo final
   python src/generate_report.py --output reporte_ejecutivo.html
   ```

---

## 6. Entregables y Criterios de Evaluación

* **Entregable 1 — Reporte Ejecutivo HTML:** [`reporte_ejecutivo.html`](file:///home/josorioam/programming/crecere_tech_proof/reporte_ejecutivo.html) (2 páginas estrictas, diseño visual pulido alineado con la identidad corporativa de Creceré AI, orientado a alta dirección).
* **Entregable 2 — Repositorio Público:** Código limpio, auditable, reproducible e idempotente, sin dependencias de secretos en tracking.
