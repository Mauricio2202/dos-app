"""
core.py
=======
Esquema compartido del prototipo: las nueve variables que alimentan a los
modelos, el reconocimiento de encabezados "alias" (para aceptar CSV de
distintas fuentes, como datasets estilo CIC-IDS) y el generador de datos
sintéticos de demostración.

Nada en este módulo abre sockets, captura tráfico ni se conecta a la red:
solo define estructuras de datos y transforma lo que ya está cargado en
memoria (un DataFrame de pandas).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Las nueve variables que ven los modelos. El orden importa para la
# importancia de variables que reporta XGBoost, así que lo fijamos aquí
# una sola vez y todo lo demás importa esta lista en vez de repetirla.
# ---------------------------------------------------------------------------
FEATURES: list[str] = [
    "packets_per_second",
    "bytes_per_second",
    "syn_ratio",
    "ack_ratio",
    "unique_ports",
    "active_connections",
    "mean_packet_size",
    "tcp_ratio",
    "udp_ratio",
]

# Columnas descriptivas: viajan junto a los datos para que el dashboard
# pueda mostrarlas, pero nunca se le pasan al clasificador ni al detector
# de anomalías.
DESCRIPTIVE_COLUMNS: list[str] = ["source_ip", "destination_port", "label", "timestamp"]


def _normalize_header(name: str) -> str:
    """Reduce un encabezado a una forma comparable: minúsculas, sin
    espacios/guiones/slashes. 'Flow Packets/s' y 'flow_packets_s' terminan
    siendo la misma clave, lo que es justo el propósito de esta función."""
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


# Cada clave de la derecha es un nombre "canónico" interno que usamos antes
# de calcular las nueve features. No son las features en sí: algunas
# (packet_count, duration, syn_count...) son insumos para derivarlas.
_ALIAS_MAP: dict[str, str] = {
    # packets_per_second
    "flowpacketss": "packets_per_second",
    "packetspersecond": "packets_per_second",
    "pps": "packets_per_second",
    # bytes_per_second
    "flowbytess": "bytes_per_second",
    "bytespersecond": "bytes_per_second",
    "bps": "bytes_per_second",
    # conteos para derivar tasas cuando faltan las tasas directas
    "totalfwdpackets": "packet_count",
    "totalpackets": "packet_count",
    "packetcount": "packet_count",
    "totalfwdbytes": "byte_count",
    "totallengthoffwdpackets": "byte_count",
    "bytecount": "byte_count",
    "flowduration": "duration",
    "duration": "duration",
    # flags
    "synflagcount": "syn_count",
    "syncount": "syn_count",
    "ackflagcount": "ack_count",
    "ackcount": "ack_count",
    # puertos / conexiones (para proxies)
    "destinationport": "destination_port",
    "dstport": "destination_port",
    "sourceip": "source_ip",
    "srcip": "source_ip",
    # protocolo (para tcp_ratio / udp_ratio cuando es una fila por protocolo)
    "protocol": "protocol",
    # etiqueta
    "label": "label",
    "class": "label",
    "attacktype": "label",
}


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = _normalize_header(str(col))
        if key in _ALIAS_MAP:
            rename_map[col] = _ALIAS_MAP[key]
    return df.rename(columns=rename_map)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """División elemento a elemento que nunca produce inf/NaN: donde el
    denominador es 0 (o el resultado no es finito), el resultado es 0.
    Así no hace falta sembrar el código de chequeos manuales después."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(0.0)


def normalize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Convierte un DataFrame con encabezados heterogéneos en uno que trae
    las nueve FEATURES (más las columnas descriptivas que existan).

    Reglas, en el orden en que se aplican:
    1. Si existen packets_per_second / bytes_per_second, se usan tal cual.
    2. Si faltan pero hay packet_count/byte_count y duration, se derivan
       dividiendo conteo entre duración.
    3. syn_ratio y ack_ratio = syn_count/ack_count divididos entre packet_count.
    4. mean_packet_size = bytes_per_second / packets_per_second.
    5. Si no hay forma de calcular unique_ports o active_connections, se
       usan proxies: la presencia de un puerto por fila como unique_ports,
       y el conteo de paquetes como active_connections.
    6. tcp_ratio / udp_ratio: si hay una columna 'protocol' por fila, se
       convierte en un indicador binario (1.0 si es TCP/UDP, 0.0 si no).
    7. Cualquier valor faltante, infinito o inválido en las nueve features
       termina en 0.
    """
    df = _apply_aliases(raw.copy())
    n = len(df)
    out = pd.DataFrame(index=df.index)

    has = lambda c: c in df.columns  # noqa: E731

    # --- tasas (paquetes/s, bytes/s) ---------------------------------
    if has("packets_per_second"):
        out["packets_per_second"] = pd.to_numeric(df["packets_per_second"], errors="coerce")
    elif has("packet_count") and has("duration"):
        out["packets_per_second"] = _safe_divide(
            pd.to_numeric(df["packet_count"], errors="coerce"),
            pd.to_numeric(df["duration"], errors="coerce"),
        )
    else:
        out["packets_per_second"] = 0.0

    if has("bytes_per_second"):
        out["bytes_per_second"] = pd.to_numeric(df["bytes_per_second"], errors="coerce")
    elif has("byte_count") and has("duration"):
        out["bytes_per_second"] = _safe_divide(
            pd.to_numeric(df["byte_count"], errors="coerce"),
            pd.to_numeric(df["duration"], errors="coerce"),
        )
    else:
        out["bytes_per_second"] = 0.0

    # --- ratios de flags ----------------------------------------------
    packet_count = pd.to_numeric(df["packet_count"], errors="coerce") if has("packet_count") else pd.Series(0.0, index=df.index)
    if has("syn_count"):
        out["syn_ratio"] = _safe_divide(pd.to_numeric(df["syn_count"], errors="coerce"), packet_count)
    else:
        out["syn_ratio"] = 0.0
    if has("ack_count"):
        out["ack_ratio"] = _safe_divide(pd.to_numeric(df["ack_count"], errors="coerce"), packet_count)
    else:
        out["ack_ratio"] = 0.0

    # --- tamaño medio de paquete ---------------------------------------
    out["mean_packet_size"] = _safe_divide(out["bytes_per_second"], out["packets_per_second"])

    # --- proxies cuando no hay mejor fuente -----------------------------
    if has("destination_port"):
        out["unique_ports"] = pd.to_numeric(df["destination_port"], errors="coerce").notna().astype(float)
    else:
        out["unique_ports"] = 0.0

    if has("packet_count"):
        out["active_connections"] = packet_count
    else:
        out["active_connections"] = 0.0

    # --- indicador binario de protocolo ---------------------------------
    if has("protocol"):
        proto = df["protocol"].astype(str).str.strip().str.lower()
        out["tcp_ratio"] = (proto == "tcp").astype(float)
        out["udp_ratio"] = (proto == "udp").astype(float)
    else:
        out["tcp_ratio"] = 0.0
        out["udp_ratio"] = 0.0

    # --- limpieza final: nada de NaN/inf en las nueve features ----------
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- columnas descriptivas, si existen -------------------------------
    for col in DESCRIPTIVE_COLUMNS:
        if col in df.columns:
            if col == "label":
                out[col] = df[col].astype(str).str.strip()
            else:
                out[col] = df[col]

    return out


@dataclass
class DemoDataSpec:
    """Parámetros del generador sintético. Separarlos de la función hace
    explícito qué es reproducible (la semilla) y qué se podría ajustar
    (filas por clase) sin tocar la lógica de generación."""

    seed: int = 42
    rows_per_class: int = 120
    classes: tuple[str, ...] = field(
        default_factory=lambda: ("Normal", "SYN flood", "HTTP flood", "Slow connections")
    )


def demo_data(spec: DemoDataSpec | None = None) -> pd.DataFrame:
    """Genera el conjunto sintético de demostración: 120 filas por clase
    (480 en total con las cuatro clases por defecto), reproducible con
    semilla 42. Los perfiles son deliberadamente separables -- el informe
    técnico es explícito en que esto sirve para probar gráficas y modelos,
    no como evidencia de detección real."""
    spec = spec or DemoDataSpec()
    rng = np.random.default_rng(spec.seed)

    # Centros de cada clase para packets_per_second y bytes_per_second;
    # el resto de variables se deriva de forma consistente con el perfil.
    profiles = {
        "Normal": dict(pps=(50, 15), bps=(40_000, 10_000), syn=(0.05, 0.02), ack=(0.45, 0.05), tcp=0.7),
        "SYN flood": dict(pps=(4000, 800), bps=(300_000, 60_000), syn=(0.9, 0.05), ack=(0.02, 0.01), tcp=0.98),
        "HTTP flood": dict(pps=(1200, 300), bps=(900_000, 150_000), syn=(0.2, 0.05), ack=(0.5, 0.05), tcp=0.95),
        "Slow connections": dict(pps=(8, 3), bps=(2_000, 500), syn=(0.1, 0.03), ack=(0.4, 0.05), tcp=0.9),
    }

    frames = []
    base_time = pd.Timestamp("2026-01-01 00:00:00")
    for idx, cls in enumerate(spec.classes):
        prof = profiles.get(cls, profiles["Normal"])
        n = spec.rows_per_class
        pps = np.clip(rng.normal(prof["pps"][0], prof["pps"][1], n), 0, None)
        bps = np.clip(rng.normal(prof["bps"][0], prof["bps"][1], n), 0, None)
        syn_ratio = np.clip(rng.normal(prof["syn"][0], prof["syn"][1], n), 0, 1)
        ack_ratio = np.clip(rng.normal(prof["ack"][0], prof["ack"][1], n), 0, 1)
        tcp_ratio = (rng.random(n) < prof["tcp"]).astype(float)
        udp_ratio = 1.0 - tcp_ratio
        unique_ports = rng.integers(1, 50, n).astype(float)
        active_connections = np.clip(rng.normal(pps.mean(), pps.std() + 1, n), 0, None)
        mean_packet_size = _safe_divide(pd.Series(bps), pd.Series(pps)).to_numpy()

        frame = pd.DataFrame(
            {
                "packets_per_second": pps,
                "bytes_per_second": bps,
                "syn_ratio": syn_ratio,
                "ack_ratio": ack_ratio,
                "unique_ports": unique_ports,
                "active_connections": active_connections,
                "mean_packet_size": mean_packet_size,
                "tcp_ratio": tcp_ratio,
                "udp_ratio": udp_ratio,
                "label": cls,
                "timestamp": base_time + pd.to_timedelta(rng.integers(0, 3600, n), unit="s") + pd.Timedelta(minutes=15 * idx),
                # IP de documentación (RFC 5737, TEST-NET-1): nunca se resuelve
                # ni se usa para conectar a nada, solo para que el dashboard
                # tenga algo que mostrar en la columna de origen.
                "source_ip": [f"192.0.2.{rng.integers(1, 254)}" for _ in range(n)],
                "destination_port": rng.integers(1, 65535, n),
                "cpu_pct": np.clip(rng.normal(30 if cls == "Normal" else 70, 10, n), 0, 100),
                "ram_pct": np.clip(rng.normal(40 if cls == "Normal" else 75, 10, n), 0, 100),
                "latency_ms": np.clip(rng.normal(20 if cls == "Normal" else 300, 15, n), 0, None),
                "packet_loss_pct": np.clip(rng.normal(0.1 if cls == "Normal" else 5, 2, n), 0, 100),
            }
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    return data.sample(frac=1.0, random_state=spec.seed).reset_index(drop=True)
