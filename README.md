# Creceré AI — Technical Challenge: Human vs. AI Voice Agents Analysis

Este repositorio contiene el pipeline analítico para la evaluación comparativa del desempeño de agentes de cobranza y gestión humana vs. agentes de voz basados en Inteligencia Artificial (Voicebots).

## Estructura del Proyecto

```text
├── src/
│   ├── transcriber.py           # Ingesta y transcripción/diarización con Deepgram nova-3
│   ├── diarization_parser.py    # Procesamiento y formateo de turnos de diálogo
│   ├── feature_extractor.py     # Extracción estructurada de variables (acústicas y semánticas)
│   ├── statistical_analysis.py  # Modelado estadístico, tests de hipótesis y econometría
│   └── generate_report.py       # Generador autónomo del reporte ejecutivo en HTML (máx 2 págs)
├── reporte_ejecutivo.html       # Reporte final interactivo y listo para impresión A4
├── .gitignore                   # Exclusión estricta de audios, transcripciones y secretos
├── requirements.txt             # Dependencias del proyecto
└── README.md
```

## Requisitos y Configuración

1. Clonar el repositorio.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configurar credenciales en `.env` (siguiendo `.env.example`).

## Pipeline de Ejecución

1. **Transcripción y Diarización Acústica:**
   ```bash
   python src/transcriber.py
   python src/diarization_parser.py
   ```

2. **Extracción Semántica Estructurada:**
   ```bash
   python src/feature_extractor.py
   ```

3. **Análisis Estadístico y Modelado Econométrico:**
   ```bash
   python src/statistical_analysis.py
   ```

4. **Generación del Reporte Ejecutivo en HTML (2 Páginas):**
   ```bash
   python src/generate_report.py --output reporte_ejecutivo.html
   ```
