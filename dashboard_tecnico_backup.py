"""
Dashboard de monitoramento climático — GNN (grafo v0.8.0).

Carrega os artefatos salvos por `grafo-0.8.0-pyg.ipynb` (pasta `artifacts/`) e
simula o monitoramento das estações de São Paulo, exibindo grafo, séries,
anomalias (Regra 1 + anomaly score de Mahalanobis), estimativa dos nós cegos
(Feature Propagation + GraphSAGE) e métricas do modelo.

Uso:
    streamlit run dashboard.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

st.set_page_config(page_title="Monitoramento Climático GNN", layout="wide")


# ---------------------------------------------------------------------------
# Carregamento dos artefatos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def carregar_artefatos():
    def _load_pkl(nome):
        with open(os.path.join(ARTIFACTS_DIR, nome), "rb") as f:
            return pickle.load(f)

    graph_data = _load_pkl("graph_data.pkl")
    tau = _load_pkl("tau.pkl")
    sigma_inv = _load_pkl("sigma_inv.pkl")
    test = _load_pkl("X_test.pkl")
    return graph_data, tau, sigma_inv, test


if not os.path.isdir(ARTIFACTS_DIR) or not os.path.exists(os.path.join(ARTIFACTS_DIR, "graph_data.pkl")):
    st.error(
        "Artefatos não encontrados em `artifacts/`. "
        "Execute o notebook `grafo-0.8.0-pyg.ipynb` por completo para gerá-los."
    )
    st.stop()

graph_data, tau, sigma_inv, test = carregar_artefatos()

nodes_lista = graph_data["nodes_lista"]
coords = graph_data["coords"]
labels = graph_data["labels"]
blind_nodes = set(graph_data["blind_nodes"])
feature_names = graph_data["feature_names"]
edge_index = np.asarray(graph_data["edge_index"])

X_2024 = np.asarray(test["X_2024"])                  # [T, N, F]
X_2024_baseline = np.asarray(test["X_2024_baseline"])
timestamps = pd.to_datetime(test["timestamps_2024"])
X_hat_mv = np.asarray(test["X_hat_mv"])              # [T-1, N, F]
X_real_next = np.asarray(test["X_real_next"])        # [T-1, N, F]
alert_confirmado = np.asarray(test["alert_confirmado"])  # [T-1, N, F] bool
anomaly_scores = np.asarray(test["anomaly_scores"])  # [T-1, N]
df_alertas = test.get("df_alertas", pd.DataFrame())

N_NODES = len(nodes_lista)
N_FEATS = len(feature_names)
T_PRED = X_hat_mv.shape[0]

# Timestamps das previsões (t+1).
ts_pred = timestamps[1:1 + T_PRED]


def idx_no(node_str):
    return nodes_lista.index(node_str)


def label_no(node_str):
    return f"{node_str} ({labels.get(node_str, node_str)})"


st.title("Monitoramento Climático com GNN — São Paulo")
st.caption(
    "Feature Propagation (Rossi et al., 2022) + GraphSAGE | "
    "Regra 1 (limiar previsto, P95) + Regra 2 (Mahalanobis)"
)

# Seletor de timestep para o estado de monitoramento "atual".
t_sel = st.sidebar.slider(
    "Timestep de monitoramento (previsão t+h)",
    min_value=0, max_value=max(T_PRED - 1, 0), value=min(T_PRED - 1, 0),
)
st.sidebar.write("Instante:", ts_pred[t_sel] if T_PRED else "-")
st.sidebar.write("Estações:", N_NODES, "| Nós cegos:", sorted(blind_nodes))

# Alerta ativo por nó no timestep selecionado (qualquer variável).
alerta_no_t = alert_confirmado[t_sel].any(axis=1) if T_PRED else np.zeros(N_NODES, bool)


tab_grafo, tab_mon, tab_as, tab_cegos, tab_metrics = st.tabs(
    ["Grafo", "Monitoramento", "Anomaly Score", "Nós Cegos", "Métricas do Modelo"]
)


# ---------------------------------------------------------------------------
# Aba 1 — Grafo
# ---------------------------------------------------------------------------
with tab_grafo:
    st.subheader("Mapa das estações")
    rows = []
    for n in nodes_lista:
        i = idx_no(n)
        lat, lon = coords.get(n, (0.0, 0.0))
        if n in blind_nodes:
            estado = "Nó cego"
        elif alerta_no_t[i]:
            estado = "Alerta"
        else:
            estado = "Normal"
        valores = {feature_names[f]: float(X_hat_mv[t_sel, i, f]) if T_PRED else float(X_2024[0, i, f])
                   for f in range(N_FEATS)}
        hover = "<br>".join(f"{k}: {v:.2f}" for k, v in valores.items())
        rows.append({
            "no": n, "label": labels.get(n, n), "lat": lat, "lon": lon,
            "estado": estado, "hover": hover,
        })
    dfm = pd.DataFrame(rows)

    cmap = {"Normal": "green", "Alerta": "red", "Nó cego": "gray"}

    # Arestas do grafo (linhas).
    edge_lat, edge_lon = [], []
    for e in range(edge_index.shape[1]):
        a, b = int(edge_index[0, e]), int(edge_index[1, e])
        la, lo = coords.get(nodes_lista[a], (0.0, 0.0))
        lb, lob = coords.get(nodes_lista[b], (0.0, 0.0))
        edge_lat += [la, lb, None]
        edge_lon += [lo, lob, None]

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=edge_lat, lon=edge_lon, mode="lines",
        line=dict(width=0.5, color="lightgray"), hoverinfo="skip", name="arestas",
    ))
    for estado, cor in cmap.items():
        sub = dfm[dfm["estado"] == estado]
        if sub.empty:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=sub["lat"], lon=sub["lon"], mode="markers",
            marker=dict(size=12, color=cor),
            text=[f"<b>{label_no(n)}</b><br>{h}" for n, h in zip(sub["no"], sub["hover"])],
            hoverinfo="text", name=estado,
        ))
    if not dfm.empty:
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=9,
            mapbox_center={"lat": dfm["lat"].mean(), "lon": dfm["lon"].mean()},
            height=600, margin=dict(l=0, r=0, t=0, b=0),
        )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Verde = normal | Vermelho = alerta ativo | Cinza = nó cego")


# ---------------------------------------------------------------------------
# Aba 2 — Monitoramento
# ---------------------------------------------------------------------------
with tab_mon:
    st.subheader("Série histórica + previsão")
    c1, c2 = st.columns(2)
    est = c1.selectbox("Estação", nodes_lista, format_func=label_no)
    var = c2.selectbox("Variável", feature_names)
    i = idx_no(est)
    f = feature_names.index(var)

    JAN = 144  # 24h a cada 10 min
    fim = min(t_sel + 1, T_PRED)
    ini = max(0, fim - JAN)

    tempo_hist = ts_pred[ini:fim]
    serie_real = X_real_next[ini:fim, i, f]
    serie_prev = X_hat_mv[ini:fim, i, f]

    mes_atual = int(ts_pred[t_sel].month) if T_PRED else 1
    # banda P5-P95 do mês atual (P95 vem de tau; P5 aproximado por simetria de mínimos)
    p95 = float(tau[mes_atual - 1, i, f])
    p5 = float(np.percentile(X_2024[:, i, f], 5))

    fig = go.Figure()
    if len(tempo_hist):
        fig.add_hrect(y0=p5, y1=p95, fillcolor="LightSkyBlue", opacity=0.25,
                      line_width=0, annotation_text="P5–P95 (mês)")
        fig.add_trace(go.Scatter(x=tempo_hist, y=serie_real, mode="lines",
                                 name="Real (t+1)", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=tempo_hist, y=serie_prev, mode="lines",
                                 name="Previsto (modelo)", line=dict(color="royalblue", dash="dash")))
        # marcadores de alerta Regra 1
        alert_mask = alert_confirmado[ini:fim, i, f]
        if alert_mask.any():
            fig.add_trace(go.Scatter(
                x=np.asarray(tempo_hist)[alert_mask],
                y=serie_prev[alert_mask], mode="markers",
                marker=dict(color="red", size=9, symbol="x"),
                name="Alerta Regra 1",
            ))
    fig.update_layout(height=480, xaxis_title="Tempo", yaxis_title=var,
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Banda mês {mes_atual:02d}: P5={p5:.2f}, P95={p95:.2f}")


# ---------------------------------------------------------------------------
# Aba 3 — Anomaly Score
# ---------------------------------------------------------------------------
with tab_as:
    st.subheader("Alertas e Anomaly Score (Mahalanobis)")
    if isinstance(df_alertas, pd.DataFrame) and not df_alertas.empty:
        st.dataframe(
            df_alertas.sort_values("AS_Score", ascending=False).reset_index(drop=True),
            use_container_width=True, height=320,
        )
    else:
        st.info("Nenhum alerta confirmado pela Regra 1 no conjunto de teste.")

    st.markdown("#### Heatmap de Anomaly Score por estação × variável")
    # AS por (nó, variável): contribuição |e_k| / sqrt(sigma_kk) no timestep t_sel.
    if T_PRED:
        e = X_real_next[t_sel] - X_hat_mv[t_sel]  # [N, F]
        sig_diag = np.array([np.diag(sigma_inv[i]) for i in range(N_NODES)])  # [N, F]
        contrib = np.abs(e) * np.sqrt(np.clip(sig_diag, 0, None))  # [N, F]
        heat = pd.DataFrame(contrib, index=nodes_lista, columns=feature_names)
        figh = px.imshow(heat, aspect="auto", color_continuous_scale="Reds",
                         labels=dict(x="Variável", y="Estação", color="AS contrib."))
        figh.update_layout(height=700)
        st.plotly_chart(figh, use_container_width=True)
        st.caption(f"Anomaly score total no instante: máx={anomaly_scores[t_sel].max():.2f}")


# ---------------------------------------------------------------------------
# Aba 4 — Nós Cegos
# ---------------------------------------------------------------------------
with tab_cegos:
    st.subheader("Nós cegos: baseline (média vizinhos) vs FP + GNN")
    var_c = st.selectbox("Variável", feature_names, key="var_cego")
    f = feature_names.index(var_c)

    for n in sorted(blind_nodes):
        if n not in nodes_lista:
            continue
        i = idx_no(n)
        st.markdown(f"**Nó cego {label_no(n)}**")
        serie_fp = X_2024[:, i, f]
        serie_base = X_2024_baseline[:, i, f]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps[:len(serie_fp)], y=serie_fp,
                                 mode="lines", name="FP + GNN (estimado)",
                                 line=dict(color="royalblue")))
        fig.add_trace(go.Scatter(x=timestamps[:len(serie_base)], y=serie_base,
                                 mode="lines", name="Baseline (média vizinhos)",
                                 line=dict(color="orange", dash="dot")))
        fig.update_layout(height=320, xaxis_title="Tempo", yaxis_title=var_c,
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        mae = float(np.mean(np.abs(serie_fp - serie_base)))
        rmse = float(np.sqrt(np.mean((serie_fp - serie_base) ** 2)))
        m1, m2 = st.columns(2)
        m1.metric(f"MAE (FP vs baseline) — {var_c}", f"{mae:.3f}")
        m2.metric(f"RMSE (FP vs baseline) — {var_c}", f"{rmse:.3f}")


# ---------------------------------------------------------------------------
# Aba 5 — Métricas do Modelo
# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Métricas do modelo no conjunto de teste (2024)")

    df_res_chuva = test.get("df_resultados_chuva")
    df_res_temp = test.get("df_resultados_temp")
    resumo_pred = test.get("resumo_pred")
    df_metricas_cegos = test.get("df_metricas_cegos")

    if resumo_pred is not None:
        st.markdown("#### Resumo médio por variável (FP + GraphSAGE)")
        st.dataframe(resumo_pred, use_container_width=True)

    if df_metricas_cegos is not None:
        st.markdown("#### Nós cegos: baseline (média simples) vs FP + GraphSAGE")
        st.dataframe(df_metricas_cegos, use_container_width=True)

    cols = st.columns(2)
    if df_res_chuva is not None:
        cols[0].markdown("#### Pluviosidade — MAE/RMSE por nó")
        cols[0].dataframe(df_res_chuva[["no", "mae", "rmse"]].sort_values("rmse"),
                          use_container_width=True, height=400)
    if df_res_temp is not None:
        cols[1].markdown("#### Temperatura — MAE/RMSE por nó")
        cols[1].dataframe(df_res_temp[["no", "mae", "rmse"]].sort_values("rmse"),
                          use_container_width=True, height=400)
