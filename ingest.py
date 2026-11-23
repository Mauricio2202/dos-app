"""
ingest.py
=========
Lectura de datos desde archivos ya cargados en memoria (bytes): CSV y
capturas PCAP/PCAPNG. Nada aquí abre una interfaz de red, transmite
paquetes ni hace captura en vivo -- solo parsea bytes que el dashboard ya
recibió (por ejemplo, de un `st.file_uploader`).

Por qué bytes y no una ruta de archivo: así el dashboard puede pasar
exactamente lo que subió la persona sin tener que escribirlo a disco
primero, y `pcap_to_flows` puede recibir la misma captura sin que importe
si Streamlit la entregó como un objeto en memoria.
"""

from __future__ import annotations

import io
from collections import defaultdict

import pandas as pd

import core

MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MiB, igual que el límite del dashboard
MAX_PACKETS = 200_000
WINDOW_SECONDS = 10.0
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"


class FileTooLargeError(ValueError):
    """El archivo supera el límite de 200 MiB que el dashboard documenta."""


class InsufficientLabelsError(ValueError):
    """No hay al menos dos clases con al menos dos filas cada una, que es
    lo mínimo que XGBoost necesita para una evaluación supervisada con
    partición estratificada."""


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Carga un CSV completo en memoria con pandas y aplica la
    normalización de `core`. No hay procesamiento por bloques: el archivo
    se materializa entero, de ahí el límite de tamaño."""
    if len(file_bytes) > MAX_FILE_BYTES:
        mib = len(file_bytes) / (1024 * 1024)
        raise FileTooLargeError(
            f"El archivo pesa {mib:.1f} MiB; el límite del dashboard es 200 MiB."
        )
    raw = pd.read_csv(io.BytesIO(file_bytes))
    return core.normalize_dataframe(raw)


def validate_supervised_labels(df: pd.DataFrame) -> None:
    """Revisa el requisito mínimo para entrenar el clasificador: columna
    'label' con al menos dos clases distintas y al menos dos filas por
    clase (para que la partición estratificada 75/25 tenga algo que
    repartir en cada lado)."""
    if "label" not in df.columns:
        raise InsufficientLabelsError("El archivo no trae una columna de etiqueta ('label').")
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise InsufficientLabelsError("Se necesitan al menos dos clases distintas en 'label'.")
    if (counts < 2).any():
        pocas = ", ".join(counts[counts < 2].index.astype(str))
        raise InsufficientLabelsError(f"Estas clases tienen menos de dos filas: {pocas}.")


# ---------------------------------------------------------------------------
# PCAP / PCAPNG
# ---------------------------------------------------------------------------

def _open_reader(file_bytes: bytes):
    """Elige PcapReader o PcapNgReader según el magic number del archivo
    ya cargado en memoria. Se importa Scapy aquí (no al tope del módulo)
    para que el resto de ingest.py funcione aunque Scapy -- y su
    dependencia opcional de libpcap -- no esté instalado; esa ausencia
    solo afecta la captura en vivo, que este prototipo no implementa."""
    from scapy.utils import PcapNgReader, PcapReader

    buf = io.BytesIO(file_bytes)
    magic = buf.read(4)
    buf.seek(0)
    if magic == _PCAPNG_MAGIC:
        return PcapNgReader(buf)
    return PcapReader(buf)


def pcap_to_flows(
    file_bytes: bytes,
    max_packets: int = MAX_PACKETS,
    window_seconds: float = WINDOW_SECONDS,
) -> tuple[pd.DataFrame, dict]:
    """Agrega una captura PCAP/PCAPNG por IP de origen y ventana fija de
    `window_seconds`. Solo procesa paquetes IPv4 (IPv6 se ignora); no
    reconstruye conexiones TCP, no reensambla sesiones y no agrega de
    forma bidireccional -- cada paquete cuenta una sola vez, en el grupo
    de su propia IP de origen.

    Devuelve (DataFrame con las nueve features + columnas descriptivas,
    metadata) donde metadata indica cuántos paquetes se procesaron y si
    se alcanzó el tope de `max_packets` (es decir, si la captura se
    truncó).
    """
    from scapy.layers.inet import IP, TCP, UDP

    groups: dict[tuple[str, int], dict] = defaultdict(
        lambda: {
            "packet_count": 0,
            "byte_count": 0,
            "syn_count": 0,
            "ack_count": 0,
            "tcp_count": 0,
            "udp_count": 0,
            "dest_ports": set(),
            "window_start": None,
        }
    )

    processed = 0
    first_ts: float | None = None
    truncated = False

    reader = _open_reader(file_bytes)
    try:
        for pkt in reader:
            if processed >= max_packets:
                truncated = True
                break
            processed += 1

            if IP not in pkt:
                continue  # sin IPv4 no hay por dónde agrupar

            ts = float(pkt.time)
            if first_ts is None:
                first_ts = ts
            window_idx = int((ts - first_ts) // window_seconds)
            src = pkt[IP].src
            g = groups[(src, window_idx)]
            if g["window_start"] is None:
                g["window_start"] = first_ts + window_idx * window_seconds

            g["packet_count"] += 1
            g["byte_count"] += len(pkt)

            dst_port = None
            if TCP in pkt:
                g["tcp_count"] += 1
                flags = int(pkt[TCP].flags)
                if flags & 0x02:  # bandera SYN
                    g["syn_count"] += 1
                if flags & 0x10:  # bandera ACK
                    g["ack_count"] += 1
                dst_port = int(pkt[TCP].dport)
            elif UDP in pkt:
                g["udp_count"] += 1
                dst_port = int(pkt[UDP].dport)

            if dst_port is not None:
                g["dest_ports"].add(dst_port)
    finally:
        reader.close()

    rows = []
    for (src, _window_idx), g in groups.items():
        pc = g["packet_count"]
        pps = pc / window_seconds
        bps = g["byte_count"] / window_seconds
        rows.append(
            {
                "source_ip": src,
                "timestamp": pd.to_datetime(g["window_start"], unit="s", errors="coerce"),
                "packets_per_second": pps,
                "bytes_per_second": bps,
                "syn_ratio": (g["syn_count"] / pc) if pc else 0.0,
                "ack_ratio": (g["ack_count"] / pc) if pc else 0.0,
                "unique_ports": float(len(g["dest_ports"])),
                "active_connections": float(len(g["dest_ports"])),
                "mean_packet_size": (bps / pps) if pps else 0.0,
                "tcp_ratio": (g["tcp_count"] / pc) if pc else 0.0,
                "udp_ratio": (g["udp_count"] / pc) if pc else 0.0,
            }
        )

    df = pd.DataFrame(rows, columns=["source_ip", "timestamp", *core.FEATURES])
    for col in core.FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    metadata = {
        "packets_processed": processed,
        "truncated": truncated,
        "flows": len(df),
    }
    return df, metadata
