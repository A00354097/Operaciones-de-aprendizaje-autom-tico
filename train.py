#!/usr/bin/env python3
"""
train.py
Script modular de entrenamiento parametrizado para clasificación de riesgo de diabetes.
Instrumentado con MLFlow Tracking.
Materia: Operaciones de aprendizaje automático (TC5061).
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def parse_args():
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Pipeline parametrizado de Regresión Logística con MLFlow Tracking para Diabetes Risk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hiperparámetro 1: Inverso de regularización C
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización (menor valor implica regularización más fuerte)",
    )
    # Hiperparámetro 2: Algoritmo de optimización
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "saga", "newton-cg"],
        help="Algoritmo de optimización para la pérdida logística multinomial",
    )
    # Hiperparámetro 3: Iteraciones máximas
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones de convergencia del optimizador",
    )
    # Hiperparámetro 4: Ponderación de balance de clases
    parser.add_argument(
        "--class_weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Estrategia de compensación de pesos para clases desbalanceadas",
    )
    # Semilla aleatoria para reproducibilidad
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Semilla aleatoria fija para la partición y la optimización",
    )
    # Ruta del dataset
    parser.add_argument(
        "--data_path",
        type=str,
        default="diabetes_risk.csv",
        help="Ruta relativa o absoluta hacia el archivo diabetes_risk.csv",
    )
    # Nombre del experimento en MLflow
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="TC5061_Diabetes_Risk_Tracking",
        help="Nombre del contenedor de experimentos en MLflow",
    )
    return parser.parse_args()


def load_and_split_data(data_path: str, random_state: int, test_size: float = 0.20):
    """Carga los datos, elimina identificadores irrelevantes y efectúa partición estratificada."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {data_path}")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["patient_id", "diabetes_risk"])
    y = df["diabetes_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(X_sample: pd.DataFrame, C: float, solver: str, max_iter: int, class_weight: str, random_state: int):
    """Construye un ColumnTransformer y acopla el clasificador de regresión logística."""
    num_cols = X_sample.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_sample.select_dtypes(include=["object", "string"]).columns.tolist()

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])

    cw = None if class_weight == "none" else "balanced"

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight=cw,
            random_state=random_state,
        )),
    ])
    return model_pipeline


def evaluate_and_generate_artifacts(model, X_test, y_test, output_dir="artifacts"):
    """Calcula métricas globales y multiclase, y almacena la matriz de confusión y el reporte JSON."""
    os.makedirs(output_dir, exist_ok=True)
    labels = ["Low", "Moderate", "High"]
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1_m = float(f1_score(y_test, y_pred, average="macro"))
    f1_w = float(f1_score(y_test, y_pred, average="weighted"))

    recalls = recall_score(y_test, y_pred, labels=labels, average=None)
    precisions = precision_score(y_test, y_pred, labels=labels, average=None)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "recall_low_risk": float(recalls[0]),
        "recall_moderate_risk": float(recalls[1]),
        "recall_high_risk": float(recalls[2]),
        "precision_low_risk": float(precisions[0]),
        "precision_moderate_risk": float(precisions[1]),
        "precision_high_risk": float(precisions[2]),
    }

    # Artefacto 1: Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(f"Matriz de Confusión (F1-Macro: {f1_m:.4f})", fontsize=12, pad=12)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    # Artefacto 2: Resumen estructurado en formato JSON
    json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics, cm_path, json_path


def main():
    args = parse_args()

    # Usar ruta absoluta exacta a la base de datos SQLite
    db_path = os.path.abspath("mlruns_db.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(args.experiment_name)

    # 1. Cargar y partir datos
    X_train, X_test, y_train, y_test = load_and_split_data(
        args.data_path, args.random_state
    )

    # 2. Iniciar sesión de tracking en MLflow
    run_name = f"logit_C={args.C}_solver={args.solver}_cw={args.class_weight}"
    with mlflow.start_run(run_name=run_name):
        # A. Registrar parámetros de corrida
        mlflow.log_param("C", args.C)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("test_split_ratio", 0.20)

        # B. Construir y ajustar el pipeline
        pipeline = build_pipeline(
            X_train,
            args.C,
            args.solver,
            args.max_iter,
            args.class_weight,
            args.random_state,
        )
        pipeline.fit(X_train, y_train)

        # C. Evaluar modelo y exportar artefactos
        metrics, cm_path, json_path = evaluate_and_generate_artifacts(
            pipeline, X_test, y_test
        )

        # D. Registrar métricas en MLflow
        for m_name, m_val in metrics.items():
            mlflow.log_metric(m_name, m_val)

        # E. Registrar artefactos
        mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(json_path, artifact_path="evaluation_metrics")

        # F. Registrar pipeline completo como modelo de MLflow con firma
        signature = infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_train.head(3),
        )

        print(f"Corrida finalizada exitosamente [{run_name}]")
        print(f" -> Accuracy: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f} | Recall High: {metrics['recall_high_risk']:.4f}")


if __name__ == "__main__":
    main()