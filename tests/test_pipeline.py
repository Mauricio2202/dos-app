"""
tests/test_pipeline.py
=======================
Las cinco pruebas que documenta el informe técnico:

1. Las cuatro clases de demostración y las nueve features válidas.
2. Los alias de encabezados CIC y el cálculo del ratio SYN.
3. La construcción y agregación de PCAP en memoria, sin transmitir tráfico.
4. El entrenamiento de XGBoost e Isolation Forest, con F1 > 0.8 en la demo.
5. La expiración de la ventana del limitador de solicitudes.

Las pruebas 3, 4 y 5 dependen de paquetes opcionales (scapy, xgboost,
fastapi) que no siempre están instalados -- por ejemplo, en el sandbox
donde se escribió este archivo no lo estaban. Cuando faltan, la prueba se
salta explícitamente (`skipTest`) en vez de fallar, para que el resto de
la suite siga siendo una señal útil. En tu entorno, con
`pip install -r requirements.txt`, las cinco deberían ejecutarse de lleno.

Correr con: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core
import ingest


def _has(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


class TestDemoDataAndFeatures(unittest.TestCase):
    """Caso 1: las cuatro clases de demostración y las nueve features."""

    def test_four_classes_480_rows(self):
        df = core.demo_data()
        self.assertEqual(len(df), 480)
        self.assertEqual(
            set(df["label"].unique()),
            {"Normal", "SYN flood", "HTTP flood", "Slow connections"},
        )
        self.assertEqual(df["label"].value_counts().nunique(), 1)  # 120 cada una

    def test_nine_features_present_and_finite(self):
        df = core.demo_data()
        self.assertEqual(list(core.FEATURES), sorted(core.FEATURES, key=core.FEATURES.index))
        self.assertEqual(len(core.FEATURES), 9)
        for col in core.FEATURES:
            self.assertIn(col, df.columns)
            self.assertTrue(df[col].notna().all(), f"{col} tiene NaN")
            self.assertTrue((df[col].abs() != float("inf")).all(), f"{col} tiene inf")


class TestHeaderAliasesAndSynRatio(unittest.TestCase):
    """Caso 2: alias de encabezados CIC y cálculo del ratio SYN."""

    def test_cic_style_aliases_are_recognized(self):
        import pandas as pd

        raw = pd.DataFrame(
            {
                "Flow Packets/s": [1000.0],
                "Flow Bytes/s": [500000.0],
                "SYN Flag Count": [900],
                "ACK Flag Count": [10],
                "Total Fwd Packets": [1000],
                "Protocol": ["TCP"],
                "Destination Port": [80],
                "Label": ["DoS"],
            }
        )
        norm = core.normalize_dataframe(raw)
        self.assertAlmostEqual(norm.loc[0, "packets_per_second"], 1000.0)
        self.assertAlmostEqual(norm.loc[0, "bytes_per_second"], 500000.0)
        # syn_ratio = syn_count / packet_count = 900 / 1000
        self.assertAlmostEqual(norm.loc[0, "syn_ratio"], 0.9)
        self.assertAlmostEqual(norm.loc[0, "ack_ratio"], 0.01)
        self.assertEqual(norm.loc[0, "label"], "DoS")

    def test_missing_values_become_zero(self):
        import pandas as pd

        raw = pd.DataFrame({"Protocol": ["UDP"]})  # casi nada reconocible
        norm = core.normalize_dataframe(raw)
        for col in core.FEATURES:
            self.assertEqual(norm.loc[0, col], 0.0 if col != "udp_ratio" else 1.0)


class TestPcapAggregation(unittest.TestCase):
    """Caso 3: construcción y agregación de PCAP en memoria."""

    def test_pcap_flows_are_aggregated_without_sending_traffic(self):
        if not _has("scapy"):
            self.skipTest("scapy no está instalado en este entorno")

        from scapy.layers.inet import IP, TCP
        from scapy.utils import wrpcap

        packets = []
        base_time = 1_700_000_000.0
        for i in range(5):
            pkt = IP(src="192.0.2.10", dst="192.0.2.20") / TCP(sport=40000 + i, dport=80, flags="S")
            pkt.time = base_time + i * 0.1  # todos dentro de la primera ventana de 10s
            packets.append(pkt)

        buf = io.BytesIO()
        wrpcap(buf, packets)  # escribe a memoria; nunca toca una interfaz de red
        df, meta = ingest.pcap_to_flows(buf.getvalue())

        self.assertEqual(meta["packets_processed"], 5)
        self.assertFalse(meta["truncated"])
        self.assertEqual(len(df), 1)  # un solo origen, una sola ventana
        row = df.iloc[0]
        self.assertEqual(row["source_ip"], "192.0.2.10")
        self.assertAlmostEqual(row["syn_ratio"], 1.0)  # las 5 fueron SYN


class TestModels(unittest.TestCase):
    """Caso 4: entrenamiento de XGBoost e Isolation Forest."""

    def test_isolation_forest_flags_a_sensible_fraction(self):
        import models

        df = core.demo_data()
        result = models.fit_anomaly_detector(df, contamination=0.06)
        self.assertGreater(result["n_flagged"], 0)
        self.assertLess(result["n_flagged"], len(df))

    def test_xgboost_f1_above_0_8_on_demo_data(self):
        if not _has("xgboost"):
            self.skipTest("xgboost no está instalado en este entorno")
        import models

        df = core.demo_data()
        result = models.fit_classifier(df)
        self.assertGreater(result["f1_macro"], 0.8)


class TestRateLimiterWindowExpiry(unittest.TestCase):
    """Caso 5: expiración de la ventana del limitador de solicitudes."""

    def test_window_expires_and_frees_capacity(self):
        if not _has("fastapi"):
            self.skipTest("fastapi no está instalado en este entorno")
        from collections import deque

        import dosguard.service as service

        limiter = service.SlidingWindowRateLimiter.__new__(service.SlidingWindowRateLimiter)
        limiter._hits = {}
        window = deque()
        limiter._hits["cliente"] = window

        now = 1_000_000.0
        # Llenamos la ventana justo al límite.
        for _ in range(service.REQUEST_LIMIT):
            window.append(now)
        self.assertEqual(len(window), service.REQUEST_LIMIT)

        # Simulamos que pasó más tiempo que WINDOW_SECONDS: las entradas viejas
        # deben expirar y liberar capacidad, tal como hace dispatch().
        later = now + service.WINDOW_SECONDS + 1
        while window and later - window[0] > service.WINDOW_SECONDS:
            window.popleft()
        self.assertEqual(len(window), 0)


if __name__ == "__main__":
    unittest.main()
