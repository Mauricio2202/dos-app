"""
risk.py
=======
Un semáforo de riesgo instantáneo, calculado sin entrenar ningún modelo:
un puntaje robusto (MAD, no desviación estándar, para que un solo pico
extremo no dispare falsos positivos en cascada) sobre packets_per_second
y bytes_per_second. Sirve para la pestaña de Gerencia (una lectura en
segundos) y como encabezado compartido en la vista de SOC -- no sustituye
a Isolation Forest/XGBoost, es la señal "¿hay algo raro ahora mismo?"
antes de que alguien decida entrenar un modelo completo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_COLORS = {
    "Normal": "#22c55e",
    "Elevado": "#f59e0b",
    "Crítico": "#ef4444",
}

RISK_RECOMMENDATIONS = {
    "Normal": "No se requiere acción inmediata. Seguir monitoreando.",
    "Elevado": "Se recomienda que el equipo de seguridad revise el tráfico reciente.",
    "Crítico": "Escalar al equipo de respuesta a incidentes de inmediato.",
}


@dataclass
class RiskResult:
    level: str
    color: str
    pct_outliers: float
    recommendation: str
    is_outlier: pd.Series
    top_sources: pd.DataFrame | None


def _robust_z(series: pd.Series) -> pd.Series:
    """Puntaje tipo z, pero con mediana y MAD en vez de media y desviación
    estándar: un solo valor absurdamente alto (como un pico de ataque real)
    no debería inflar el umbral que usamos para detectar justo eso."""
    median = series.median()
    mad = (series - median).abs().median()
    mad = mad if mad > 0 else 1e-9
    return ((series - median) / (1.4826 * mad)).abs()


def assess(df: pd.DataFrame) -> RiskResult:
    z_pps = _robust_z(df["packets_per_second"])
    z_bps = _robust_z(df["bytes_per_second"])
    is_outlier = (z_pps > 3.5) | (z_bps > 3.5)
    pct = float(is_outlier.mean() * 100)

    if pct < 2:
        level = "Normal"
    elif pct < 8:
        level = "Elevado"
    else:
        level = "Crítico"

    top_sources = None
    if "source_ip" in df.columns and is_outlier.any():
        top_sources = (
            df.loc[is_outlier, "source_ip"]
            .value_counts()
            .head(5)
            .rename_axis("origen")
            .reset_index(name="eventos")
        )

    return RiskResult(
        level=level,
        color=RISK_COLORS[level],
        pct_outliers=pct,
        recommendation=RISK_RECOMMENDATIONS[level],
        is_outlier=is_outlier,
        top_sources=top_sources,
    )
