"""
app.py
======
Dashboard Streamlit con dos vistas, elegidas en la barra lateral:

- "SOC (analista)": las seis pestañas técnicas completas (Resumen,
  Distribución, Modelos, Servicio API, Topología, Datos), con un semáforo
  de riesgo arriba para triage rápido.
- "Gerencia / Dirección": una sola pantalla, en lenguaje llano, pensada
  para alguien que necesita saber "¿hay un problema o no?" sin tener que
  interpretar PPS, ratios de SYN o matrices de confusión.

Ambas vistas leen la misma fuente de datos (core/ingest/models) -- la
diferencia es cuánto y cómo se muestra, no qué se calcula.

Cómo correrlo: `streamlit run app.py --server.address 127.0.0.1 --server.port 8501`
(ver README.md para el flujo completo con entorno virtual).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core
import ingest
import models
import risk

st.set_page_config(page_title="Detección de DoS", layout="wide", page_icon="🛡️")

PLOTLY_TEMPLATE = "plotly_dark"

CUSTOM_CSS = """
<style>
.kpi-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
}
.kpi-label {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.55);
    margin-top: 4px;
}
.risk-banner {
    border-radius: 14px;
    padding: 22px 26px;
    display: flex;
    align-items: center;
    gap: 22px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 18px;
}
.risk-dot {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 0 6px rgba(255,255,255,0.06);
}
.risk-level {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
}
.risk-sub {
    color: rgba(255,255,255,0.65);
    margin: 2px 0 0 0;
    font-size: 0.95rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def risk_banner(result: risk.RiskResult, context: str = "") -> None:
    st.markdown(
        f"""
        <div class="risk-banner" style="background: {result.color}1a; border-color: {result.color}55;">
            <div class="risk-dot" style="background: {result.color};"></div>
            <div>
                <p class="risk-level" style="color: {result.color};">Riesgo: {result.level}</p>
                <p class="risk-sub">{result.pct_outliers:.1f}% del tráfico se aparta del comportamiento típico{context}. {result.recommendation}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Carga de datos y manejo de estado (igual que antes)
# ---------------------------------------------------------------------------

def _source_key(name: str, n_rows: int, columns: tuple[str, ...]) -> tuple:
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
        except Exception as exc:
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


st.sidebar.header("Vista")
view = st.sidebar.radio("¿Para quién es esta pantalla?", ["SOC (analista)", "Gerencia / Dirección"])

data, source_name = _load_source()

if data is None:
    st.title("🛡️ Detección y clasificación de ataques DoS")
    st.info("Elige una fuente de datos en la barra lateral para empezar.")
    st.stop()

key = _source_key(source_name, len(data), tuple(data.columns))
if st.session_state.get("source_key") != key:
    st.session_state["source_key"] = key
    st.session_state.pop("clf_result", None)
    st.session_state.pop("anomaly_result", None)

risk_result = risk.assess(data)


# ---------------------------------------------------------------------------
# Vista Gerencia / Dirección: una sola pantalla, lenguaje llano
# ---------------------------------------------------------------------------

def render_gerencia(df: pd.DataFrame, rr: risk.RiskResult, source: str) -> None:
    st.title("🛡️ Estado de la red")
    st.caption(f"Actualizado con: {source} · {len(df)} registros analizados")

    risk_banner(rr)

    cols = st.columns(3)
    with cols[0]:
        st.markdown(kpi_card("Nivel de riesgo", rr.level, "Según el tráfico más reciente"), unsafe_allow_html=True)
    with cols[1]:
        normal_pct = 100 - rr.pct_outliers
        st.markdown(kpi_card("Tráfico normal", f"{normal_pct:.1f}%", "Del total analizado"), unsafe_allow_html=True)
    with cols[2]:
        n_sources = df["source_ip"].nunique() if "source_ip" in df.columns else None
        value = str(n_sources) if n_sources is not None else "—"
        st.markdown(kpi_card("Orígenes distintos", value, "Direcciones de red vistas"), unsafe_allow_html=True)

    st.write("")
    st.subheader("Actividad reciente")
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        ts = df.dropna(subset=["timestamp"]).copy()
        ts["minuto"] = pd.to_datetime(ts["timestamp"]).dt.floor("min")
        serie = ts.groupby("minuto")["packets_per_second"].mean().reset_index()
        fig = px.line(serie, x="minuto", y="packets_per_second", template=PLOTLY_TEMPLATE)
        fig.update_layout(
            showlegend=False,
            xaxis_title="Hora",
            yaxis_title="Actividad de red (relativa)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_traces(line_color=rr.color)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Esta fuente no trae marcas de tiempo para graficar la tendencia.")

    if rr.top_sources is not None and not rr.top_sources.empty:
        st.subheader("Orígenes bajo observación")
        st.caption("Direcciones con más eventos fuera de lo normal en este periodo.")
        st.dataframe(rr.top_sources, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "Esta pantalla resume la señal para una decisión rápida. "
        "El detalle técnico (modelos, matrices de confusión, captura de paquetes) "
        "está en la vista \"SOC (analista)\", en el menú de la izquierda."
    )


# ---------------------------------------------------------------------------
# Vista SOC: las seis pestañas técnicas
# ---------------------------------------------------------------------------

def render_soc(df: pd.DataFrame, rr: risk.RiskResult, source: str) -> None:
    st.title("🛡️ Detección y clasificación de ataques DoS")
    st.caption(f"Fuente activa: {source} · {len(df)} filas")

    risk_banner(rr, context=" (umbral estadístico, no sustituye el entrenamiento de un modelo)")

    tabs = st.tabs(["Resumen", "Distribución", "Modelos", "Servicio API", "Topología", "Datos"])

    # -- Pestaña 1: Resumen -------------------------------------------------
    with tabs[0]:
        cols = st.columns(4)
        cols[0].markdown(kpi_card("Filas", f"{len(df):,}"), unsafe_allow_html=True)
        if "label" in df.columns:
            n_labels = df["label"].nunique()
            no_normal = (df["label"].str.lower() != "normal").mean() * 100
            cols[1].markdown(kpi_card("Etiquetas distintas", str(n_labels)), unsafe_allow_html=True)
            cols[2].markdown(kpi_card("Proporción no-normal", f"{no_normal:.1f}%"), unsafe_allow_html=True)
        cols[3].markdown(kpi_card("Mediana PPS", f"{df['packets_per_second'].median():.1f}"), unsafe_allow_html=True)

        st.write("")
        if rr.top_sources is not None and not rr.top_sources.empty:
            st.subheader("Top orígenes de riesgo")
            st.dataframe(rr.top_sources, use_container_width=True, hide_index=True)

        if "timestamp" in df.columns and df["timestamp"].notna().any():
            st.subheader("Paquetes por segundo a lo largo del tiempo")
            ts = df.dropna(subset=["timestamp"]).copy()
            ts["minuto"] = pd.to_datetime(ts["timestamp"]).dt.floor("min")
            serie = ts.groupby("minuto")["packets_per_second"].mean().reset_index()
            st.plotly_chart(
                px.line(serie, x="minuto", y="packets_per_second", title="PPS promedio por minuto", template=PLOTLY_TEMPLATE),
                use_container_width=True,
            )

        if "label" in df.columns:
            col_a, col_b = st.columns(2)
            with col_a:
                counts = df["label"].value_counts().reset_index()
                counts.columns = ["label", "conteo"]
                st.plotly_chart(
                    px.bar(counts, x="label", y="conteo", title="Conteo por etiqueta", template=PLOTLY_TEMPLATE),
                    use_container_width=True,
                )
            with col_b:
                st.plotly_chart(
                    px.scatter(
                        df, x="packets_per_second", y="bytes_per_second", color="label",
                        title="Dispersión PPS vs bytes/s por clase", opacity=0.7, template=PLOTLY_TEMPLATE,
                    ),
                    use_container_width=True,
                )

    # -- Pestaña 2: Distribución ---------------------------------------------
    with tabs[1]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                px.histogram(df, x="packets_per_second", nbins=40, title="Histograma de PPS", template=PLOTLY_TEMPLATE),
                use_container_width=True,
            )
        with col_b:
            corr = df[core.FEATURES].corr()
            st.plotly_chart(
                px.imshow(corr, text_auto=".2f", title="Correlación entre variables", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=PLOTLY_TEMPLATE),
                use_container_width=True,
            )

        if "source_ip" in df.columns:
            st.subheader("Doce principales fuentes (por paquetes/s acumulado)")
            top = (
                df.groupby("source_ip")["packets_per_second"]
                .sum()
                .sort_values(ascending=False)
                .head(12)
                .reset_index()
            )
            st.plotly_chart(px.bar(top, x="source_ip", y="packets_per_second", template=PLOTLY_TEMPLATE), use_container_width=True)

        op_cols = [c for c in ("cpu_pct", "ram_pct", "latency_ms", "packet_loss_pct") if c in df.columns]
        if op_cols:
            st.subheader("Métricas operativas (si la fuente las aporta)")
            melted = df[op_cols].melt(var_name="métrica", value_name="valor")
            st.plotly_chart(px.box(melted, x="métrica", y="valor", template=PLOTLY_TEMPLATE), use_container_width=True)
        else:
            st.caption("La fuente actual no trae CPU/RAM/latencia/pérdida; no se estiman.")

    # -- Pestaña 3: Modelos ---------------------------------------------------
    with tabs[2]:
        mode = st.radio("Tipo de modelo", ["Supervisado (XGBoost)", "No supervisado (Isolation Forest)"], horizontal=True)

        if mode == "Supervisado (XGBoost)":
            if st.button("Entrenar clasificador"):
                try:
                    ingest.validate_supervised_labels(df)
                    with st.spinner("Entrenando XGBoost..."):
                        st.session_state["clf_result"] = models.fit_classifier(df)
                except ingest.InsufficientLabelsError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.error(str(exc))

            result = st.session_state.get("clf_result")
            if result:
                m = st.columns(4)
                m[0].markdown(kpi_card("Accuracy", f"{result['accuracy']:.3f}"), unsafe_allow_html=True)
                m[1].markdown(kpi_card("Precisión macro", f"{result['precision_macro']:.3f}"), unsafe_allow_html=True)
                m[2].markdown(kpi_card("Recall macro", f"{result['recall_macro']:.3f}"), unsafe_allow_html=True)
                m[3].markdown(kpi_card("F1 macro", f"{result['f1_macro']:.3f}"), unsafe_allow_html=True)
                st.caption(f"Entrenamiento: {result['n_train']} filas · Prueba: {result['n_test']} filas")

                col_a, col_b = st.columns(2)
                with col_a:
                    cm_labels = result["confusion_matrix_labels"]
                    st.plotly_chart(
                        px.imshow(
                            result["confusion_matrix"], x=cm_labels, y=cm_labels,
                            text_auto=True, title="Matriz de confusión",
                            labels=dict(x="Predicho", y="Real"), template=PLOTLY_TEMPLATE,
                        ),
                        use_container_width=True,
                    )
                with col_b:
                    importance = pd.DataFrame(
                        sorted(result["feature_importance"].items(), key=lambda kv: kv[1]),
                        columns=["variable", "importancia"],
                    )
                    st.plotly_chart(
                        px.bar(importance, x="importancia", y="variable", orientation="h", title="Importancia de variables", template=PLOTLY_TEMPLATE),
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
                        st.session_state["anomaly_result"] = models.fit_anomaly_detector(df, contamination=contamination)
                except ValueError as exc:
                    st.error(str(exc))

            result = st.session_state.get("anomaly_result")
            if result:
                st.markdown(kpi_card("Filas marcadas como anómalas", f"{result['n_flagged']} de {len(df)}"), unsafe_allow_html=True)
                plot_df = df[core.FEATURES].copy()
                plot_df["clasificación"] = result["classification"]
                plot_df["score"] = result["scores"]
                st.plotly_chart(
                    px.scatter(
                        plot_df, x="packets_per_second", y="bytes_per_second",
                        color="clasificación", title="Dispersión de anomalías",
                        hover_data=["score"], opacity=0.7, template=PLOTLY_TEMPLATE,
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

    # -- Pestaña 4: Servicio API -----------------------------------------------
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
            m[0].markdown(kpi_card("Solicitudes", f"{total:,}"), unsafe_allow_html=True)
            m[1].markdown(kpi_card("HTTP 429", f"{rejected} ({rejected / total * 100:.1f}%)"), unsafe_allow_html=True)
            m[2].markdown(kpi_card("Latencia p95", f"{p95:.1f} ms"), unsafe_allow_html=True)

            events_df["timestamp_utc"] = pd.to_datetime(events_df["timestamp_utc"], errors="coerce")
            per_min = events_df.set_index("timestamp_utc").resample("1min").size().reset_index(name="solicitudes")
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(px.line(per_min, x="timestamp_utc", y="solicitudes", title="Solicitudes por minuto", template=PLOTLY_TEMPLATE), use_container_width=True)
            with col_b:
                by_status = events_df["status"].value_counts().reset_index()
                by_status.columns = ["status", "conteo"]
                st.plotly_chart(px.bar(by_status, x="status", y="conteo", title="Códigos de respuesta HTTP", template=PLOTLY_TEMPLATE), use_container_width=True)
            st.plotly_chart(px.histogram(events_df, x="latency_ms", nbins=40, title="Histograma de latencia", template=PLOTLY_TEMPLATE), use_container_width=True)
        else:
            st.info("Todavía no hay eventos del API que mostrar.")

    # -- Pestaña 5: Topología ---------------------------------------------------
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

        fig.update_layout(title="Topología de referencia", xaxis=dict(visible=False), yaxis=dict(visible=False), height=420, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    # -- Pestaña 6: Datos --------------------------------------------------------
    with tabs[5]:
        missing = df[core.FEATURES].isna().sum()
        summary = {
            "fuente": source,
            "filas": len(df),
            "features": core.FEATURES,
            "faltantes_por_feature": {k: int(v) for k, v in missing.items()},
        }
        st.subheader("Resumen en JSON")
        st.json(summary)

        st.subheader("Primeras 100 filas normalizadas")
        st.dataframe(df.head(100), use_container_width=True)

        st.download_button(
            "Descargar CSV normalizado",
            df.to_csv(index=False).encode("utf-8"),
            file_name="datos_normalizados.csv",
            mime="text/csv",
        )


if view == "Gerencia / Dirección":
    render_gerencia(data, risk_result, source_name)
else:
    render_soc(data, risk_result, source_name)
