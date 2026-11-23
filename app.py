"""
app.py
======
Dashboard Streamlit: la interfaz que une core.py (esquema y datos
sintéticos), ingest.py (CSV/PCAP) y models.py (XGBoost + Isolation
Forest) en seis pestañas, tal como las describe el informe técnico.

Cómo correrlo: `streamlit run app.py --server.address 127.0.0.1 --server.port 8501`
(ver README.md para el flujo completo con entorno virtual).
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core
import ingest
import models

st.set_page_config(page_title="Detección de DoS", layout="wide")


# ---------------------------------------------------------------------------
# Carga de datos y manejo de estado
# ---------------------------------------------------------------------------

def _source_key(name: str, n_rows: int, columns: tuple[str, ...]) -> tuple:
    """Lo que identifica a una fuente de datos para efectos de invalidar
    resultados guardados: nombre, número de filas y columnas. Si cualquiera
    cambia, los resultados de entrenamientos anteriores ya no aplican."""
    return (name, n_rows, columns)


def _load_source() -> tuple[pd.DataFrame | None, str]:
    st.sidebar.header("Fuente de datos")
    choice = st.sidebar.radio(
        "¿Qué quieres analizar?",
        ["Datos de demostración", "Archivo (CSV)", "Captura (PCAP/PCAPNG)"],
    )

    if choice == "Datos de demostración":
        return core.demo_data(), "demo"

    if choice == "Archivo (CSV)":
        uploaded = st.sidebar.file_uploader("Sube un CSV", type=["csv"])
        if uploaded is None:
            return None, "sin_archivo"
        try:
            df = ingest.load_csv(uploaded.getvalue())
        except ingest.FileTooLargeError as exc:
            st.sidebar.error(str(exc))
            return None, "error"
        except Exception as exc:  # error genérico de lectura, como pide el informe
            st.sidebar.error(f"No se pudo leer el archivo: {exc}")
            return None, "error"
        return df, uploaded.name

    uploaded = st.sidebar.file_uploader("Sube una captura", type=["pcap", "pcapng"])
    if uploaded is None:
        return None, "sin_archivo"
    try:
        df, meta = ingest.pcap_to_flows(uploaded.getvalue())
    except Exception as exc:
        st.sidebar.error(f"No se pudo procesar la captura: {exc}")
        return None, "error"
    st.sidebar.caption(
        f"{meta['packets_processed']} paquetes procesados, {meta['flows']} flujos."
        + (" (captura truncada al tope de 200 000 paquetes)" if meta["truncated"] else "")
    )
    return df, uploaded.name


data, source_name = _load_source()

if data is None:
    st.title("Detección y clasificación de ataques DoS")
    st.info("Elige una fuente de datos en la barra lateral para empezar.")
    st.stop()

key = _source_key(source_name, len(data), tuple(data.columns))
if st.session_state.get("source_key") != key:
    # La fuente cambió (nombre, filas o columnas): cualquier resultado de
    # un entrenamiento anterior ya no corresponde a estos datos.
    st.session_state["source_key"] = key
    st.session_state.pop("clf_result", None)
    st.session_state.pop("anomaly_result", None)

st.title("Detección y clasificación de ataques DoS")
st.caption(f"Fuente activa: {source_name} · {len(data)} filas")

tabs = st.tabs(["Resumen", "Distribución", "Modelos", "Servicio API", "Topología", "Datos"])


# ---------------------------------------------------------------------------
# Pestaña 1: Resumen
# ---------------------------------------------------------------------------
with tabs[0]:
    cols = st.columns(4)
    cols[0].metric("Filas", len(data))
    if "label" in data.columns:
        n_labels = data["label"].nunique()
        no_normal = (data["label"].str.lower() != "normal").mean() * 100
        cols[1].metric("Etiquetas distintas", n_labels)
        cols[2].metric("Proporción no-normal", f"{no_normal:.1f}%")
    cols[3].metric("Mediana PPS", f"{data['packets_per_second'].median():.1f}")

    if "timestamp" in data.columns and data["timestamp"].notna().any():
        st.subheader("Paquetes por segundo a lo largo del tiempo")
        ts = data.dropna(subset=["timestamp"]).copy()
        ts["minuto"] = pd.to_datetime(ts["timestamp"]).dt.floor("min")
        serie = ts.groupby("minuto")["packets_per_second"].mean().reset_index()
        st.plotly_chart(
            px.line(serie, x="minuto", y="packets_per_second", title="PPS promedio por minuto"),
            use_container_width=True,
        )

    if "label" in data.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            counts = data["label"].value_counts().reset_index()
            counts.columns = ["label", "conteo"]
            st.plotly_chart(
                px.bar(counts, x="label", y="conteo", title="Conteo por etiqueta"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                px.scatter(
                    data, x="packets_per_second", y="bytes_per_second", color="label",
                    title="Dispersión PPS vs bytes/s por clase", opacity=0.7,
                ),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Pestaña 2: Distribución
# ---------------------------------------------------------------------------
with tabs[1]:
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            px.histogram(data, x="packets_per_second", nbins=40, title="Histograma de PPS"),
            use_container_width=True,
        )
    with col_b:
        corr = data[core.FEATURES].corr()
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", title="Correlación entre variables", color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
            use_container_width=True,
        )

    if "source_ip" in data.columns:
        st.subheader("Doce principales fuentes (por paquetes/s acumulado)")
        top = (
            data.groupby("source_ip")["packets_per_second"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        st.plotly_chart(px.bar(top, x="source_ip", y="packets_per_second"), use_container_width=True)

    op_cols = [c for c in ("cpu_pct", "ram_pct", "latency_ms", "packet_loss_pct") if c in data.columns]
    if op_cols:
        st.subheader("Métricas operativas (si la fuente las aporta)")
        melted = data[op_cols].melt(var_name="métrica", value_name="valor")
        st.plotly_chart(px.box(melted, x="métrica", y="valor"), use_container_width=True)
    else:
        st.caption("La fuente actual no trae CPU/RAM/latencia/pérdida; no se estiman.")


# ---------------------------------------------------------------------------
# Pestaña 3: Modelos
# ---------------------------------------------------------------------------
with tabs[2]:
    mode = st.radio("Tipo de modelo", ["Supervisado (XGBoost)", "No supervisado (Isolation Forest)"], horizontal=True)

    if mode == "Supervisado (XGBoost)":
        if st.button("Entrenar clasificador"):
            try:
                ingest.validate_supervised_labels(data)
                with st.spinner("Entrenando XGBoost..."):
                    st.session_state["clf_result"] = models.fit_classifier(data)
            except ingest.InsufficientLabelsError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))

        result = st.session_state.get("clf_result")
        if result:
            m = st.columns(4)
            m[0].metric("Accuracy", f"{result['accuracy']:.3f}")
            m[1].metric("Precisión macro", f"{result['precision_macro']:.3f}")
            m[2].metric("Recall macro", f"{result['recall_macro']:.3f}")
            m[3].metric("F1 macro", f"{result['f1_macro']:.3f}")
            st.caption(f"Entrenamiento: {result['n_train']} filas · Prueba: {result['n_test']} filas")

            col_a, col_b = st.columns(2)
            with col_a:
                cm_labels = result["confusion_matrix_labels"]
                st.plotly_chart(
                    px.imshow(
                        result["confusion_matrix"], x=cm_labels, y=cm_labels,
                        text_auto=True, title="Matriz de confusión",
                        labels=dict(x="Predicho", y="Real"),
                    ),
                    use_container_width=True,
                )
            with col_b:
                importance = pd.DataFrame(
                    sorted(result["feature_importance"].items(), key=lambda kv: kv[1]),
                    columns=["variable", "importancia"],
                )
                st.plotly_chart(
                    px.bar(importance, x="importancia", y="variable", orientation="h", title="Importancia de variables"),
                    use_container_width=True,
                )

            st.download_button(
                "Descargar predicciones de prueba (CSV)",
                result["predictions"].to_csv(index=False).encode("utf-8"),
                file_name="predicciones_prueba.csv",
                mime="text/csv",
            )

    else:
        contamination = st.slider("Contaminación esperada", 0.01, 0.25, 0.06, 0.01)
        if st.button("Ejecutar Isolation Forest"):
            try:
                with st.spinner("Ejecutando Isolation Forest..."):
                    st.session_state["anomaly_result"] = models.fit_anomaly_detector(data, contamination=contamination)
            except ValueError as exc:
                st.error(str(exc))

        result = st.session_state.get("anomaly_result")
        if result:
            st.metric("Filas marcadas como anómalas", f"{result['n_flagged']} de {len(data)}")
            plot_df = data[core.FEATURES].copy()
            plot_df["clasificación"] = result["classification"]
            plot_df["score"] = result["scores"]
            st.plotly_chart(
                px.scatter(
                    plot_df, x="packets_per_second", y="bytes_per_second",
                    color="clasificación", title="Dispersión de anomalías",
                    hover_data=["score"], opacity=0.7,
                ),
                use_container_width=True,
            )
            st.caption("La contaminación es un umbral configurado, no una medida real de cuántos ataques hay. Revisa los casos marcados antes de actuar sobre ellos.")
            st.download_button(
                "Descargar resultado (CSV)",
                plot_df.to_csv(index=False).encode("utf-8"),
                file_name="anomalias.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Pestaña 4: Servicio API
# ---------------------------------------------------------------------------
with tabs[3]:
    st.caption(
        "Corre el servicio de demostración por separado "
        "(`uvicorn dosguard.service:app --host 127.0.0.1 --port 8000`) y súbelo aquí; "
        "Streamlit no lo inicia ni lo consulta automáticamente."
    )
    events_file = st.file_uploader("data/http_events.csv", type=["csv"], key="http_events")
    events_df = None
    if events_file is not None:
        events_df = pd.read_csv(events_file)
    else:
        default_path = __import__("pathlib").Path(__file__).resolve().parent / "data" / "http_events.csv"
        if default_path.exists():
            events_df = pd.read_csv(default_path)
            st.caption(f"Usando {default_path}")

    if events_df is not None and not events_df.empty:
        total = len(events_df)
        rejected = (events_df["status"] == 429).sum()
        p95 = events_df["latency_ms"].quantile(0.95)
        m = st.columns(3)
        m[0].metric("Solicitudes", total)
        m[1].metric("HTTP 429", f"{rejected} ({rejected / total * 100:.1f}%)")
        m[2].metric("Latencia p95 (ms)", f"{p95:.1f}")

        events_df["timestamp_utc"] = pd.to_datetime(events_df["timestamp_utc"], errors="coerce")
        per_min = events_df.set_index("timestamp_utc").resample("1min").size().reset_index(name="solicitudes")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.line(per_min, x="timestamp_utc", y="solicitudes", title="Solicitudes por minuto"), use_container_width=True)
        with col_b:
            by_status = events_df["status"].value_counts().reset_index()
            by_status.columns = ["status", "conteo"]
            st.plotly_chart(px.bar(by_status, x="status", y="conteo", title="Códigos de respuesta HTTP"), use_container_width=True)
        st.plotly_chart(px.histogram(events_df, x="latency_ms", nbins=40, title="Histograma de latencia"), use_container_width=True)
    else:
        st.info("Todavía no hay eventos del API que mostrar.")


# ---------------------------------------------------------------------------
# Pestaña 5: Topología
# ---------------------------------------------------------------------------
with tabs[4]:
    st.caption(
        "Reconstrucción documental de la topología de referencia -- no es una "
        "topología importada de ningún archivo .pkt; sirve para ubicar dónde "
        "encajaría cada pieza del prototipo en una red real."
    )

    nodes = {
        "clientes": (0.0, 1.0, "Clientes Wi-Fi"),
        "ap": (1.2, 1.0, "Access Point"),
        "switch": (2.4, 1.0, "Switch"),
        "firewall": (3.6, 1.0, "Firewall"),
        "router": (4.8, 1.0, "Router"),
        "servidor": (2.4, 0.0, "Servidor"),
        "ids": (3.6, 0.0, "IDS / Observatorio"),
    }
    edges = [("clientes", "ap"), ("ap", "switch"), ("switch", "firewall"), ("firewall", "router"), ("switch", "servidor")]
    span_edge = ("switch", "ids")

    fig = go.Figure()
    for a, b in edges:
        ax, ay, _ = nodes[a]
        bx, by, _ = nodes[b]
        fig.add_trace(go.Scatter(x=[ax, bx], y=[ay, by], mode="lines", line=dict(color="#888", width=2), hoverinfo="none", showlegend=False))

    ax, ay, _ = nodes[span_edge[0]]
    bx, by, _ = nodes[span_edge[1]]
    fig.add_trace(go.Scatter(x=[ax, bx], y=[ay, by], mode="lines", line=dict(color="#d62728", width=2, dash="dash"), name="SPAN (espejo de tráfico)"))

    for key_, (x, y, label) in nodes.items():
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[label], textposition="bottom center", marker=dict(size=28, color="#1f77b4"), showlegend=False))

    fig.update_layout(title="Topología de referencia", xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Pestaña 6: Datos
# ---------------------------------------------------------------------------
with tabs[5]:
    missing = data[core.FEATURES].isna().sum()
    summary = {
        "fuente": source_name,
        "filas": len(data),
        "features": core.FEATURES,
        "faltantes_por_feature": {k: int(v) for k, v in missing.items()},
    }
    st.subheader("Resumen en JSON")
    st.json(summary)

    st.subheader("Primeras 100 filas normalizadas")
    st.dataframe(data.head(100), use_container_width=True)

    st.download_button(
        "Descargar CSV normalizado",
        data.to_csv(index=False).encode("utf-8"),
        file_name="datos_normalizados.csv",
        mime="text/csv",
    )
