"""Item 2 do prof.: estende o mascaramento artificial comparando o Feature
Propagation com krigagem ordinaria e interpolacao linear por triangulacao
(o analogo correto de 'bilinear' para estacoes irregulares), alem de MV e IDW.

FP/MV/IDW reconstroem a estacao mascarada a partir dos VIZINHOS DO GRAFO;
krigagem e linear usam o CAMPO COMPLETO de estacoes observadas (interpoladores
espaciais padrao). Assim testamos se o arcabouco em grafo supera a interpolacao
espacial classica na reconstrucao. Subamostra temporal para a krigagem (variograma
por instante e caro); linear e barato e usa a mesma subamostra p/ comparabilidade.
"""
import pickle, os, warnings
import numpy as np
import torch
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging

warnings.filterwarnings("ignore")
ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]; coords = g["coords"]; blind = set(g["blind_nodes"])
obs = [i for i in range(35) if nl[i] not in blind]
feat = g["feature_names"]

X = torch.tensor(d["X_2024"], dtype=torch.float32)
ei = torch.tensor(g["edge_index"]); ew = torch.tensor(g["edge_weight"], dtype=torch.float32)
N = 35; FP_ITER = 40
src, dst = ei[0], ei[1]

# subamostra temporal: 1 instante a cada 24 (2x/dia) -> ~730 instantes
T = X.shape[0]
sub = torch.arange(0, T, 24)
Xs = X[sub]; Ts = Xs.shape[0]

# coordenadas (lon, lat) em graus para os interpoladores espaciais
lon = np.array([coords[nl[i]][1] for i in range(N)])
lat = np.array([coords[nl[i]][0] for i in range(N)])


def fp_rec(Xin, known):
    Xc = Xin.clone(); Xk = Xc[:, known, :].clone(); ewb = ew.view(1, -1, 1)
    for _ in range(FP_ITER):
        msgs = Xc[:, src, :] * ewb
        Xn = torch.zeros_like(Xc); Xn.index_add_(1, dst, msgs)
        Xn[:, known, :] = Xk; Xc = Xn
    return Xc


def viz_do(alvo):
    return src[dst == alvo]


res = {m: {f: [] for f in feat} for m in ["FP", "MV", "IDW", "Krig", "Linear"]}

for alvo in obs:
    real = Xs[:, alvo, :].numpy()                          # gabarito [Ts,4]
    known = torch.ones(N, dtype=torch.bool); known[alvo] = False
    for b in blind:
        known[nl.index(b)] = False
    # --- metodos por vizinho do grafo ---
    Xhat = fp_rec(Xs, known)[:, alvo, :].numpy()
    viz = viz_do(alvo)
    mv = Xs[:, viz, :].mean(dim=1).numpy()
    w = ew[dst == alvo]; w = (w / w.sum()).view(1, -1, 1)
    idw = (Xs[:, viz, :] * w).sum(dim=1).numpy()
    # --- interpoladores espaciais sobre o campo completo (exceto o alvo e cegos) ---
    base = [i for i in obs if i != alvo]
    pts = np.column_stack([lon[base], lat[base]])
    krig = np.full((Ts, 4), np.nan); lin = np.full((Ts, 4), np.nan)
    for fi in range(4):
        vals_t = Xs[:, base, fi].numpy()                   # [Ts, nbase]
        # linear (triangulacao de Delaunay) - vetorizavel por instante
        for t in range(Ts):
            lin[t, fi] = griddata(pts, vals_t[t], (lon[alvo], lat[alvo]), method="linear")
        # krigagem ordinaria (variograma esferico) por instante
        for t in range(Ts):
            try:
                ok = OrdinaryKriging(pts[:, 0], pts[:, 1], vals_t[t],
                                     variogram_model="spherical", verbose=False, enable_plotting=False)
                z, _ = ok.execute("points", np.array([lon[alvo]]), np.array([lat[alvo]]))
                krig[t, fi] = z[0]
            except Exception:
                krig[t, fi] = np.nan
    for m, est in [("FP", Xhat), ("MV", mv), ("IDW", idw), ("Krig", krig), ("Linear", lin)]:
        for fi, f in enumerate(feat):
            e = est[:, fi] - real[:, fi]
            ok = np.isfinite(e)
            res[m][f].append(np.abs(e[ok]).mean() if ok.any() else np.nan)
    print(f"  no {nl[alvo]:>2} ok", flush=True)

print(f"\nMascaramento com krigagem/linear: {len(obs)} estacoes, {Ts} instantes (subamostra).")
print("MAE medio sobre as estacoes-caso (menor = melhor):")
print(f"{'Variavel':>12} {'FP':>8} {'MV':>8} {'IDW':>8} {'Krig':>8} {'Linear':>8}")
for f in feat:
    row = f"{f:>12}"
    for m in ["FP", "MV", "IDW", "Krig", "Linear"]:
        row += f" {np.nanmean(res[m][f]):8.3f}"
    print(row)

import json
json.dump({m: {f: [float(x) for x in res[m][f]] for f in feat} for m in res},
          open(os.path.join(ART, "fp_mascaramento_krig.json"), "w"))
print("\nsalvo: artifacts/fp_mascaramento_krig.json")
