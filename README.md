# TC5061 - Experimentación y Tracking con MLflow: Clasificación de Riesgo de Diabetes

Este repositorio contiene la migración del pipeline exploratorio de Machine Learning (desarrollado en la semana anterior) a un script modular de Python parametrizado y ejecutable (`train.py`), instrumentado con **MLflow Tracking**.

El objetivo es permitir la experimentación sistemática, el seguimiento de métricas, el control de versiones de modelos y artefactos, y la reproducibilidad estricta de cada corrida.

---

## 📁 Estructura del Repositorio

```text
Operaciones-de-aprendizaje-autom-tico/
├── data/
│   └── diabetes_risk.csv          # Dataset clínico de 15,000 registros
├── notebooks/
│   └── TC5061_Notebook_Baseline_GomezFernando.ipynb  # Notebook conservado exclusivamente para EDA
├── artifacts/
│   ├── confusion_matrix.png       # Matriz de confusión generada
│   └── metrics_summary.json       # Resumen de métricas en formato JSON
├── train.py                       # Script modular de entrenamiento y tracking
├── requirements.txt               # Archivo de dependencias con versiones
├── mlruns_db.db                   # Almacén de base de datos SQLite para MLflow Tracking
└── README.md                      # Documentación del proyecto
