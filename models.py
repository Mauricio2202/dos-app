"""
models.py
=========
Los dos modelos del prototipo:

- fit_classifier: XGBoost supervisado multiclase (Normal / SYN flood /
  HTTP flood / Slow connections, o las clases que traiga el archivo).
- fit_anomaly_detector: Isolation Forest no supervisado, no usa etiquetas.

Ambas funciones reciben un DataFrame ya normalizado por `core.py` y
devuelven resultados listos para graficar -- ninguna de las dos entrena
nada "en segundo plano": se llaman explícitamente desde el botón del
dashboard, como pide el informe técnico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import core

CLASSIFIER_PARAMS = dict(
    n_estimators=180,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.9,
    tree_method="hist",
    eval_metric="mlogloss",
    n_jobs=-1,
)


def fit_classifier(df: pd.DataFrame, seed: int = 42) -> dict:
    """Entrena XGBoost con partición estratificada 75/25. Las etiquetas
    vacías se descartan antes de particionar porque no sirven ni para
    entrenar ni para medir. LabelEncoder convierte el texto a enteros
    (XGBoost los necesita así) y después traducimos las predicciones de
    vuelta a los nombres originales, para que la matriz de confusión y las
    descargas hablen en los mismos términos que trajo el archivo."""
    import xgboost as xgb  # import perezoso: ver nota en ingest._open_reader

    data = df.dropna(subset=["label"]).copy()
    data = data[data["label"].astype(str).str.strip() != ""]
    if data.empty:
        raise ValueError("No quedan filas con etiqueta después de descartar las vacías.")

    X = data[core.FEATURES].to_numpy()
    y_text = data["label"].astype(str).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    class_weights = _balanced_sample_weight(y_train)

    clf = xgb.XGBClassifier(random_state=seed, **CLASSIFIER_PARAMS)
    clf.fit(X_train, y_train, sample_weight=class_weights)

    y_pred = clf.predict(X_test)

    labels_sorted = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    class_names = encoder.inverse_transform(labels_sorted)

    return {
        "accuracy": float((y_pred == y_test).mean()),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm,
        "confusion_matrix_labels": list(class_names),
        "feature_importance": dict(zip(core.FEATURES, clf.feature_importances_.tolist())),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "predictions": pd.DataFrame(
            {
                "y_true": encoder.inverse_transform(y_test),
                "y_pred": encoder.inverse_transform(y_pred),
            }
        ),
    }


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Pesos de clase balanceados: cada clase pesa inversamente a su
    frecuencia, para que una clase mayoritaria no domine la función de
    pérdida. Es el equivalente manual de class_weight='balanced'."""
    classes, counts = np.unique(y, return_counts=True)
    weight_by_class = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_by_class[v] for v in y])


def fit_anomaly_detector(df: pd.DataFrame, contamination: float = 0.06, seed: int = 42) -> dict:
    """Isolation Forest no supervisado: no usa 'label' en absoluto, solo
    las nueve features. Exige al menos 10 filas porque con menos el bosque
    de aislamiento no tiene suficiente contexto para que el score
    signifique algo. `contamination` es un umbral que el dashboard expone
    como control (1%-25%, 6% por defecto) -- no es una medida real de
    cuántos ataques hay en los datos, solo cuántos puntos se marcan como
    atípicos dado ese umbral."""
    if len(df) < 10:
        raise ValueError("Isolation Forest necesita al menos 10 filas.")
    contamination = float(np.clip(contamination, 0.01, 0.25))

    X = df[core.FEATURES].to_numpy()
    model = IsolationForest(
        n_estimators=180,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    raw_pred = model.fit_predict(X)  # 1 = normal, -1 = anómalo
    scores = model.decision_function(X)  # más alto = más normal

    classification = np.where(raw_pred == -1, "Anómalo", "Normal")
    return {
        "classification": classification,
        "scores": scores,
        "n_flagged": int((raw_pred == -1).sum()),
        "contamination": contamination,
    }
