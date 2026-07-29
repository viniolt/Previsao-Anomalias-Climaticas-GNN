"""
Painel do clima em São Paulo — versão simples, para o público geral.

Mostra, em linguagem direta, o que as 35 estações meteorológicas da cidade
mediram, o que o sistema esperava para a hora seguinte e onde o comportamento
fugiu do normal. A versão técnica completa (com métricas do modelo) está em
`dashboard_tecnico_backup.py`.

Uso:
    streamlit run dashboard.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
H = 6  # horizonte: 6 passos de 10 min = 1 hora

st.set_page_config(page_title="Clima em São Paulo", layout="wide")


# ---------------------------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def carregar():
    def _pkl(nome):
        with open(os.path.join(ARTIFACTS_DIR, nome), "rb") as f:
            return pickle.load(f)
    return _pkl("graph_data.pkl"), _pkl("tau.pkl"), _pkl("X_test.pkl")


if not os.path.exists(os.path.join(ARTIFACTS_DIR, "graph_data.pkl")):
    st.error("Dados não encontrados na pasta `artifacts/`.")
    st.stop()

graph_data, tau, test = carregar()

nodes_lista = graph_data["nodes_lista"]
coords = graph_data["coords"]
labels = graph_data["labels"]
blind_nodes = set(graph_data["blind_nodes"])
feature_names = graph_data["feature_names"]
edge_index = np.asarray(graph_data["edge_index"])

X_2024 = np.asarray(test["X_2024"])                      # medições [T, N, F]
timestamps = pd.to_datetime(test["timestamps_2024"])
X_hat_mv = np.asarray(test["X_hat_mv"])                  # esperado p/ t+H
X_real_next = np.asarray(test["X_real_next"])            # medido em t+H
alert_confirmado = np.asarray(test["alert_confirmado"])  # [T-H, N, F] bool
trues_chuva = np.asarray(test["trues_chuva_all"])        # chuva acumulada em t+H

N_NODES = len(nodes_lista)
T_PRED = X_hat_mv.shape[0]
ts_alvo = timestamps[H:H + T_PRED]                       # instante a que a previsão se refere

# Nomes claros para as variáveis medidas.
NOME_VAR = {
    "PLU": "Chuva acumulada (mm)",
    "Temp": "Temperatura (°C)",
    "Umid.Rel.": "Umidade do ar (%)",
    "Vel.VT": "Vento (m/s)",
}
obs_idx = [i for i, n in enumerate(nodes_lista) if n not in blind_nodes]


def idx_no(node_str):
    return nodes_lista.index(node_str)


def nome_estacao(node_str):
    nome = labels.get(node_str, node_str).strip()
    return f"{nome} (estimada)" if node_str in blind_nodes else nome


# ---------------------------------------------------------------------------
# Cabeçalho e escolha do momento
# ---------------------------------------------------------------------------
st.title("Clima em São Paulo — painel de acompanhamento")

with st.expander("Como funciona este painel?", expanded=False):
    st.markdown(
        """
        - **35 estações meteorológicas** espalhadas pela região metropolitana medem
          chuva, temperatura, umidade e vento **a cada 10 minutos**.
        - Um sistema treinado com os anos anteriores estima **o que deve acontecer
          na próxima hora** em cada estação.
        - Quando o que acontece **foge muito do esperado** (por exemplo, uma chuva
          repentina muito acima do normal para o mês), a estação entra em **alerta**
          e aparece em vermelho no mapa.
        - Três estações (em cinza) **não têm medições próprias**: os valores delas
          são estimados a partir das estações vizinhas.
        - Os dados exibidos são do **ano de 2024**. Escolha o dia e a hora ao lado
          para navegar.
        """
    )

st.sidebar.header("Escolha o momento")
data_sel = st.sidebar.date_input(
    "Dia",
    value=pd.Timestamp("2024-01-13").date(),
    min_value=ts_alvo[0].date(),
    max_value=ts_alvo[-1].date(),
)
hora_sel = st.sidebar.slider("Hora do dia", 0, 23, 1)
minuto_sel = st.sidebar.select_slider("Minutos", options=[0, 10, 20, 30, 40, 50], value=0)

alvo = pd.Timestamp.combine(data_sel, pd.Timestamp(f"{hora_sel:02d}:{minuto_sel:02d}").time())
t_sel = int(np.clip(ts_alvo.searchsorted(alvo), 0, T_PRED - 1))
st.sidebar.write("Mostrando:", ts_alvo[t_sel].strftime("%d/%m/%Y %H:%M"))

# Estado de cada estação no instante escolhido.
alerta_no_t = alert_confirmado[t_sel].any(axis=1)

# Chuva da última hora (o quanto choveu de fato na hora que termina no instante exibido).
chuva_hora = trues_chuva[t_sel] - X_2024[t_sel, :, 0]

tab_mapa, tab_estacao, tab_dia = st.tabs(
    ["Mapa da cidade", "Acompanhar uma estação", "Resumo do dia"]
)


# ---------------------------------------------------------------------------
# Aba 1 — Mapa
# ---------------------------------------------------------------------------
with tab_mapa:
    c1, c2, c3 = st.columns(3)
    c1.metric("Estações em alerta agora", int(alerta_no_t[obs_idx].sum()))
    c2.metric("Temperatura média na cidade", f"{X_real_next[t_sel, obs_idx, 1].mean():.1f} °C")
    c3.metric("Maior chuva na última hora", f"{max(chuva_hora[obs_idx].max(), 0):.1f} mm")

    rows = []
    for n in nodes_lista:
        i = idx_no(n)
        lat, lon = coords.get(n, (0.0, 0.0))
        if n in blind_nodes:
            estado = "Sem medição própria"
        elif alerta_no_t[i]:
            estado = "Alerta"
        else:
            estado = "Normal"
        hover = (
            f"Chuva na última hora: {max(float(chuva_hora[i]), 0):.1f} mm<br>"
            f"Temperatura: {float(X_real_next[t_sel, i, 1]):.1f} °C<br>"
            f"Umidade do ar: {float(X_real_next[t_sel, i, 2]):.0f} %<br>"
            f"Vento: {float(X_real_next[t_sel, i, 3]):.1f} m/s"
        )
        rows.append({"no": n, "lat": lat, "lon": lon, "estado": estado, "hover": hover})
    dfm = pd.DataFrame(rows)

    cores = {"Normal": "green", "Alerta": "red", "Sem medição própria": "gray"}

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
        line=dict(width=0.5, color="lightgray"), hoverinfo="skip", showlegend=False,
    ))
    for estado, cor in cores.items():
        sub = dfm[dfm["estado"] == estado]
        if sub.empty:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=sub["lat"], lon=sub["lon"], mode="markers",
            marker=dict(size=12, color=cor),
            text=[f"<b>{nome_estacao(n)}</b><br>{h}" for n, h in zip(sub["no"], sub["hover"])],
            hoverinfo="text", name=estado,
        ))
    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=9,
        mapbox_center={"lat": dfm["lat"].mean(), "lon": dfm["lon"].mean()},
        height=600, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Cada ponto é uma estação meteorológica; passe o mouse para ver as medições. "
        "Verde = comportamento normal. Vermelho = alerta (medições muito fora do "
        "habitual para o mês). Cinza = estação sem medições próprias, com valores "
        "estimados a partir das vizinhas. As linhas ligam estações vizinhas."
    )


# ---------------------------------------------------------------------------
# Aba 2 — Uma estação
# ---------------------------------------------------------------------------
with tab_estacao:
    c1, c2 = st.columns(2)
    est = c1.selectbox("Estação", nodes_lista, format_func=nome_estacao)
    var = c2.selectbox("O que mostrar", feature_names, format_func=lambda v: NOME_VAR[v])
    i = idx_no(est)
    f = feature_names.index(var)

    JAN = 144  # últimas 24 horas (medições a cada 10 min)
    fim = t_sel + 1
    ini = max(0, fim - JAN)

    tempo = ts_alvo[ini:fim]
    medido = X_real_next[ini:fim, i, f]
    esperado = X_hat_mv[ini:fim, i, f]

    mes = int(ts_alvo[t_sel].month)
    p95 = float(tau[mes - 1, i, f])
    p5 = float(np.percentile(X_2024[:, i, f], 5))

    fig = go.Figure()
    fig.add_hrect(
        y0=p5, y1=p95, fillcolor="LightSkyBlue", opacity=0.25, line_width=0,
        annotation_text="faixa habitual do mês",
    )
    fig.add_trace(go.Scatter(x=tempo, y=medido, mode="lines",
                             name="Medido", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=tempo, y=esperado, mode="lines",
                             name="Esperado pelo sistema", line=dict(color="royalblue", dash="dash")))
    alertas = alert_confirmado[ini:fim, i, f]
    if alertas.any():
        fig.add_trace(go.Scatter(
            x=np.asarray(tempo)[alertas], y=esperado[alertas], mode="markers",
            marker=dict(color="red", size=9, symbol="x"), name="Alerta",
        ))
    fig.update_layout(height=460, xaxis_title="Hora", yaxis_title=NOME_VAR[var],
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    if est in blind_nodes:
        st.info(
            "Esta estação não tem medições próprias: tudo o que aparece aqui foi "
            "estimado a partir das estações vizinhas."
        )
    st.caption(
        "Últimas 24 horas até o momento escolhido. A linha preta é o que a estação "
        "mediu; a linha azul tracejada é o que o sistema esperava com uma hora de "
        "antecedência. A faixa azul-clara mostra os valores habituais para o mês: "
        "quando a linha preta sai muito dela, o comportamento é incomum. Os X "
        "vermelhos marcam os alertas."
    )


# ---------------------------------------------------------------------------
# Aba 3 — Resumo do dia
# ---------------------------------------------------------------------------
with tab_dia:
    dia = pd.Timestamp(data_sel)
    mask_dia = (ts_alvo.normalize() == dia)
    if not mask_dia.any():
        st.info("Sem dados para este dia.")
    else:
        idx_dia = np.where(mask_dia)[0]
        chuva_d = (trues_chuva[idx_dia] - X_2024[idx_dia, :, 0])[:, obs_idx]
        temp_d = X_real_next[idx_dia][:, obs_idx, 1]
        alertas_d = alert_confirmado[idx_dia].any(axis=2)[:, obs_idx]

        chuva_max = float(np.clip(chuva_d, 0, None).max())
        pos = np.unravel_index(np.argmax(chuva_d), chuva_d.shape)
        est_chuva = nome_estacao(nodes_lista[obs_idx[pos[1]]])
        n_est_alerta = int(alertas_d.any(axis=0).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Maior chuva em 1 hora", f"{chuva_max:.1f} mm", help=f"Registrada em {est_chuva}")
        c2.metric("Temperatura mínima", f"{float(temp_d.min()):.1f} °C")
        c3.metric("Temperatura máxima", f"{float(temp_d.max()):.1f} °C")
        c4.metric("Estações com algum alerta", f"{n_est_alerta} de {len(obs_idx)}")

        if chuva_max > 10:
            st.warning(
                f"**Dia com chuva forte.** Em {est_chuva}, choveu {chuva_max:.0f} mm "
                "em uma única hora — o suficiente para causar pontos de alagamento."
            )
        else:
            st.success("Dia sem registro de chuva forte (nenhuma estação passou de 10 mm em 1 hora).")

        # Chuva ao longo do dia (maior valor entre as estações, hora a hora).
        horas = ts_alvo[idx_dia]
        serie = pd.Series(np.clip(chuva_d, 0, None).max(axis=1), index=horas)
        por_hora = serie.resample("1h").max()
        figb = go.Figure(go.Bar(x=por_hora.index, y=por_hora.values, marker_color="steelblue"))
        figb.update_layout(height=320, xaxis_title="Hora do dia",
                           yaxis_title="Chuva em 1 hora (mm) — pior estação")
        st.plotly_chart(figb, use_container_width=True)
        st.caption(
            "Quanto choveu em cada hora do dia, considerando a estação mais atingida "
            "naquela hora. Barras acima de 10 mm indicam chuva forte."
        )
