import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_PAI = os.path.dirname(_HERE)                       # repo: raiz | local: codigo/
ART = os.path.join(_PAI, "artifacts")
BASE = _PAI if os.path.isdir(os.path.join(_PAI, "artigo")) else os.path.dirname(_PAI)
OUT = os.path.join(BASE, "artigo", "figs")

d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]; lab = g["labels"]

i_bp = nl.index([n for n in nl if lab[n] == "Barragem Palheiros"][0])
X = d["X_2024"]; T = d["trues_chuva_all"].shape[0]
ts = pd.to_datetime(d["timestamps_2024"])

# chuva que cai em [t, t+60min] = acumulado(t+6) - acumulado(t)
inc = d["trues_chuva_all"] - X[:T, :, 0]
prev = d["preds_chuva_all"] - X[:T, :, 0]          # previsao do incremento
AS = d["anomaly_scores"]

# O eixo x e o instante ALVO (t+6): a chuva de 13/01 00h e prevista as 12/01 23h.
# Por isso o eixo original comeca em 23h.
t_alvo = ts[6:T + 6]
dia = t_alvo.date == pd.Timestamp("2024-01-13").date()
x = t_alvo[dia]
obs_mm = np.clip(inc[dia, i_bp], 0, None)
prev_mm = np.clip(prev[dia, i_bp], 0, None)
score = AS[dia, i_bp]

print(f"Barragem Palheiros 13/01: chuva max={obs_mm.max():.1f} mm | prev no pico={prev_mm[obs_mm.argmax()]:.1f} mm")
print(f"  anomaly score: inicio={score[0]:.1f} | max={score.max():.1f} as {x[score.argmax()]:%H:%M}")

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True,
                               gridspec_kw={"height_ratios": [1.5, 1]})

ax1.plot(x, obs_mm, color="#1f77b4", lw=2.6, label="Chuva observada (próx. 60 min)")
ax1.plot(x, prev_mm, color="#e8730c", lw=2.4, ls="--", label="Chuva prevista (próx. 60 min)")
ax1.set_ylabel("Chuva 60 min (mm)", fontsize=13)
ax1.legend(loc="upper right", fontsize=12, framealpha=0.95)
ax1.set_title("Evento de 13/01/2024 — Barragem Palheiros", fontsize=16, pad=12)
ax1.tick_params(labelsize=12)

ax2.plot(x, score, color="#8c1fd4", lw=2.4)
ax2.fill_between(x, score, color="#8c1fd4", alpha=0.13)
ax2.set_ylabel("Anomaly score", fontsize=13)
ax2.set_xlabel("Horário (13/01/2024)", fontsize=13)
ax2.tick_params(labelsize=12)

ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/evento_20240113.{ext}", dpi=150, bbox_inches="tight")
print("salvo:", f"{OUT}/evento_20240113.pdf / .png")
