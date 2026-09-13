# Creceré AI — Technical Challenge: Human vs. AI Voice Agents Analysis

Este repositorio contiene el pipeline analítico para la evaluación comparativa del desempeño de agentes de cobranza y gestión humana vs. agentes de voz basados en Inteligencia Artificial (Voicebots).

## Estructura del Proyecto

```text
├── src/
│   ├── transcriber.py           # Ingesta y transcripción/diarización con Deepgram nova-3
│   ├── diarization_parser.py    # Procesamiento y formateo de turnos de diálogo
│   ├── feature_extractor.py     # Extracción estructurada de variables (acústicas y semánticas)
│   └── statistical_analysis.py  # Modelado estadístico, tests de hipótesis y econometría
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
3. Configurar credenciales según sea requerido en el entorno local.
