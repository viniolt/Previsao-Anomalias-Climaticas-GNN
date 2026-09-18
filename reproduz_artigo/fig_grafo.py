"""Figura da topologia do grafo: 35 nos e 85 arestas sobre os distritos de SP.

Complementa `fig_mapa.py`, que mostra os nos mas nao as arestas. Depende apenas
de `artifacts/graph_data.pkl` (versionado) e de `distritos.geojson` — nao
precisa de X_test.pkl nem de rodar o notebook.

Saida: <pasta do relatorio>/figs/grafo.{pdf,png}
"""
import json
import math
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPoly

from caminhos import ARTIFACTS as ART, FIGS as OUT

_HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]
coords = g["coords"]
lab = g["labels"]
blind = set(g["blind_nodes"])
ei = np.asarray(g["edge_index"])


def haversine(a, b):
    la1, lo1 = map(math.radians, a)
    la2, lo2 = map(math.radians, b)
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(d))


# arestas nao direcionadas (edge_index tem os dois sentidos)
pares = {tuple(sorted((int(ei[0, e]), int(ei[1, e])))) for e in range(ei.shape[1])}
pares = sorted(pares)
dists = np.array([haversine(coords[nl[i]], coords[nl[j]]) for i, j in pares])
graus = np.zeros(len(nl), dtype=int)
for i, j in pares:
    graus[i] += 1
    graus[j] += 1

print(f"{len(nl)} nos | {len(pares)} arestas nao direcionadas")
print(f"grau: min {graus.min()} max {graus.max()} media {graus.mean():.1f}")
print(f"distancia: min {dists.min():.1f} max {dists.max():.1f} mediana "
      f"{np.median(dists):.1f} km")

# ---------------------------------------------------------------- mapa base
geo = json.load(open(os.path.join(_HERE, "distritos.geojson")))


def aneis(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [p[0] for p in geom["coordinates"]]


plt.style.use("default")
fig, ax = plt.subplots(figsize=(5.4, 5.2))
for f in geo["features"]:
    for r in aneis(f["geometry"]):
        ax.add_patch(MplPoly(np.asarray(r), closed=True, facecolor="#f0f0ec",
                             edgecolor="#c9c9c4", lw=0.35, zorder=1))

# ---------------------------------------------------------------- arestas
w = 1.0 / np.maximum(dists, 0.1)
lw = 0.45 + 1.9 * (w - w.min()) / (w.max() - w.min())
for (i, j), width in zip(pares, lw):
    la_i, lo_i = coords[nl[i]]
    la_j, lo_j = coords[nl[j]]
    ax.plot([lo_i, lo_j], [la_i, la_j], color="#3f6fa8", lw=width, alpha=0.75,
            solid_capstyle="round", zorder=3)

# ---------------------------------------------------------------- nos
for k, n in enumerate(nl):
    la, lo = coords[n]
    if n in blind:
        ax.scatter(lo, la, s=90, c="white", marker="D", edgecolors="#333333",
                   lw=1.2, zorder=6)
    else:
        ax.scatter(lo, la, s=40, c="#1c4474", marker="o", edgecolors="white",
                   lw=0.6, zorder=6)
    ax.annotate(n, (lo, la), xytext=(0, 8.5 if n in blind else 7),
                textcoords="offset points", fontsize=6.4, ha="center",
                color="#222222", zorder=7)

anota = {
    "33": ("IAG-USP", (-16, -13)),
    "34": ("INMET Interlagos", (10, -12)),
    "35": ("INMET Mirante\nde Santana", (-9, 19)),
    "28": ("Barragem Palheiros", (10, -10)),
    "14": ("Capela do Socorro", (-13, -9)),
}
for n, (txt, off) in anota.items():
    la, lo = coords[n]
    ax.annotate(txt, (lo, la), xytext=off, textcoords="offset points",
                fontsize=9.2, weight="bold", color="#111111",
                ha="left" if off[0] > 0 else "right", va="center", zorder=8)

ax.set_xlim(-46.95, -46.36)
ax.set_ylim(-23.96, -23.39)
ax.set_xticks([]); ax.set_yticks([])
for lado in ("top", "right", "bottom", "left"):
    ax.spines[lado].set_color("#bbbbbb")
ax.set_aspect("equal", adjustable="box")

leg = [Line2D([], [], ls="", marker="o", ms=5.6, mfc="#1c4474", mec="white",
              label="Estação observada (CGE-SP)"),
       Line2D([], [], ls="", marker="D", ms=6.4, mfc="white", mec="#333333",
              label="Nó cego (IAG-USP, INMET)"),
       Line2D([], [], color="#3f6fa8", lw=1.7,
              label=r"Aresta ($w_{ij}=1/d$; espessura $\propto w_{ij}$)")]
ax.legend(handles=leg, loc="lower right", fontsize=8.4, framealpha=0.95,
          borderpad=0.4, labelspacing=0.35, handletextpad=0.5)
ax.text(0.015, 0.022,
        f"{len(nl)} nós  ·  {len(pares)} arestas\n"
        f"grau {graus.min()}–{graus.max()} (média {graus.mean():.1f})\n"
        f"{dists.min():.1f}–{dists.max():.1f} km (mediana {np.median(dists):.1f} km)",
        transform=ax.transAxes, fontsize=8.4, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=0.6,
                  alpha=0.95), zorder=9)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"grafo.{ext}"), dpi=200, bbox_inches="tight")
print("salvo:", os.path.join(OUT, "grafo.pdf"))
