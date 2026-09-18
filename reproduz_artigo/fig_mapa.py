import json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.lines import Line2D

import os
from caminhos import ARTIFACTS as ART, FIGS as OUT
os.makedirs(OUT, exist_ok=True)

d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]; lab = g["labels"]; coords = g["coords"]
blind = set(g["blind_nodes"])
obs_nodes = [n for n in nl if n not in blind]

T = d["trues_chuva_all"].shape[0]
ts = pd.to_datetime(d["timestamps_2024"])
AS = d["anomaly_scores"]
t_alvo = pd.DatetimeIndex(ts[6:T + 6])

obs_idx = [nl.index(n) for n in obs_nodes]
thr99 = np.nanpercentile(AS[:, obs_idx], 99)

k = int(np.where((t_alvo.date == pd.Timestamp("2024-01-13").date())
                 & (t_alvo.hour == 0) & (t_alvo.minute == 50))[0][0])
score_k = AS[k]
det = {n: bool(score_k[nl.index(n)] > thr99) for n in obs_nodes}
print(f"00h50 13/01: {sum(det.values())} de {len(obs_nodes)} estacoes com score > p99 ({thr99:.3f})")

# ---------- mapa base: distritos de SP ----------
geo = json.load(open(os.path.join(_HERE, "distritos.geojson")))
subs = sorted({f["properties"]["ds_subpref"] for f in geo["features"]})
pal = (list(plt.cm.Pastel1.colors) + list(plt.cm.Pastel2.colors) +
       list(plt.cm.Set3.colors) + list(plt.cm.Pastel1.colors))
cor_sub = {s: pal[i % len(pal)] for i, s in enumerate(subs)}


def aneis(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [p[0] for p in geom["coordinates"]]


def area(r):
    a = np.asarray(r)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


plt.style.use("default")
fig, ax = plt.subplots(figsize=(11.5, 13.5))

cent_sub = {}
for f in geo["features"]:
    p = f["properties"]; sp = p["ds_subpref"]
    rs = aneis(f["geometry"])
    for r in rs:
        ax.add_patch(MplPoly(np.asarray(r), closed=True, facecolor=cor_sub[sp],
                             edgecolor="#5a5a5a", lw=0.45, zorder=1))
    maior = max(rs, key=area)
    c = np.asarray(maior).mean(axis=0)
    ax.text(c[0], c[1], p["ds_nome"].title(), fontsize=3.6, color="#3a3a3a",
            ha="center", va="center", zorder=3)
    cent_sub.setdefault(sp, []).append((c, area(maior)))

for sp, lst in cent_sub.items():
    cs = np.array([c for c, _ in lst]); ws = np.array([w for _, w in lst])
    c = (cs * ws[:, None]).sum(axis=0) / ws.sum()
    ax.text(c[0], c[1], sp.replace("-", "-\n"), fontsize=5.2, weight="bold",
            color="#111111", ha="center", va="center", zorder=4)

# ---------- estacoes ----------
for n in obs_nodes:
    la, lo = coords[n]
    ax.scatter(lo, la, s=210, c="#d62728" if det[n] else "#2ca02c",
               marker="o", edgecolors="black", lw=1.1, zorder=6)
for n in sorted(blind):
    la, lo = coords[n]
    ax.scatter(lo, la, s=250, c="#9e9e9e", marker="D",
               edgecolors="black", lw=1.1, zorder=6)

anota = {
    "IAG-USP":                                   ("IAG-USP (nó cego)",            (15, 0)),
    "Barragem Palheiros":                        ("Barragem Palheiros",           (15, -4)),
    "INMET - Interlagos":                        ("INMET Interlagos\n(nó cego)",  (15, -6)),
    "INMET - Convencional (Mirante de Santana)": ("INMET Mirante de Santana\n(nó cego)", (14, 10)),
}
for n in nl:
    if lab[n] in anota:
        txt, off = anota[lab[n]]
        la, lo = coords[n]
        ax.annotate(txt, (lo, la), xytext=off, textcoords="offset points",
                    fontsize=11.5, weight="bold", va="center", zorder=7)

ax.set_xlim(-46.95, -46.30); ax.set_ylim(-24.05, -23.28)
ax.set_xlabel("Longitude", fontsize=14); ax.set_ylabel("Latitude", fontsize=14)
ax.tick_params(labelsize=12)
ax.set_aspect("equal", adjustable="box")

leg = [Line2D([], [], ls="", marker="o", ms=13, mfc="#d62728", mec="black", label="Evento detectado (anomaly > p99)"),
       Line2D([], [], ls="", marker="o", ms=13, mfc="#2ca02c", mec="black", label="Normal"),
       Line2D([], [], ls="", marker="D", ms=13, mfc="#9e9e9e", mec="black", label="Nó cego (estimado)")]
ax.legend(handles=leg, loc="lower right", fontsize=12, framealpha=0.95)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/mapa_alertas_20240113.{ext}", dpi=150, bbox_inches="tight")
print("salvo:", f"{OUT}/mapa_alertas_20240113.pdf / .png")
