"""Mascaramento artificial (pedido do prof. Ivan): em vez de validar o Feature
Propagation em um unico no cego, escondemos CADA estacao observada por vez,
reconstruimos suas features so a partir dos vizinhos e medimos o erro. Isso
produz dezenas de nos-caso reais em lugar de um.

Compara tres reconstrucoes espaciais (sem previsao temporal, um instante por vez):
  FP  = Feature Propagation (difusao D^-1 A, 40 it., ancoras restauradas)
  MV  = media simples dos vizinhos (1 passo)
  IDW = inverse distance weighting (pesos 1/Haversine, normalizados)
"""
import pickle, os
import numpy as np
import torch

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]; blind = set(g["blind_nodes"])
obs = [i for i in range(35) if nl[i] not in blind]

X = torch.tensor(d["X_2024"], dtype=torch.float32)          # [T,N,4] real nos obs
ei = torch.tensor(g["edge_index"])
ew = torch.tensor(g["edge_weight"], dtype=torch.float32)    # D^-1 A (row-stochastic)
N = 35; FP_ITER = 40
src, dst = ei[0], ei[1]

# amostra temporal (para caber na memoria e ser rapido): 1 instante por hora
T = X.shape[0]
sub = torch.arange(0, T, 6)
Xs = X[sub]                                                  # [Ts,N,4]
Ts = Xs.shape[0]


def fp_reconstruir(Xin, known_mask):
    """Difusao com restauracao de ancoras (identico ao notebook)."""
    Xc = Xin.clone()
    Xk = Xc[:, known_mask, :].clone()
    ewb = ew.view(1, -1, 1)
    for _ in range(FP_ITER):
        msgs = Xc[:, src, :] * ewb
        Xn = torch.zeros_like(Xc)
        Xn.index_add_(1, dst, msgs)
        Xn[:, known_mask, :] = Xk
        Xc = Xn
    return Xc


def mv_reconstruir(Xin, alvo):
    """Media simples dos vizinhos do no alvo (1 passo)."""
    viz = src[dst == alvo]
    return Xin[:, viz, :].mean(dim=1)                       # [Ts,4]


def idw_reconstruir(Xin, alvo):
    """Inverse distance weighting: ew ja e 1/Haversine normalizado por destino."""
    mask = dst == alvo
    viz = src[mask]; w = ew[mask]
    w = w / w.sum()
    return (Xin[:, viz, :] * w.view(1, -1, 1)).sum(dim=1)   # [Ts,4]


feat = g["feature_names"]
res = {m: {f: [] for f in feat} for m in ["FP", "MV", "IDW"]}

for alvo in obs:                                            # esconde cada obs por vez
    real = Xs[:, alvo, :].clone()                          # gabarito
    known = torch.ones(N, dtype=torch.bool)
    known[alvo] = False                                    # alvo vira "cego"
    for b in blind:
        known[nl.index(b)] = False                         # cegos reais seguem cegos
    Xhat = fp_reconstruir(Xs, known)[:, alvo, :]
    est = {"FP": Xhat, "MV": mv_reconstruir(Xs, alvo), "IDW": idw_reconstruir(Xs, alvo)}
    for m, e in est.items():
        for fi, f in enumerate(feat):
            mae = (e[:, fi] - real[:, fi]).abs().mean().item()
            res[m][f].append(mae)

print(f"Mascaramento artificial: {len(obs)} estacoes observadas escondidas, "
      f"{Ts} instantes (1/h) de 2024.\n")
print("MAE medio sobre as 32 estacoes-caso (media +- desvio entre estacoes):")
print(f"{'Variavel':>12} {'FP':>16} {'MV':>16} {'IDW':>16}")
for f in feat:
    row = f"{f:>12}"
    for m in ["FP", "MV", "IDW"]:
        a = np.array(res[m][f])
        row += f"   {a.mean():6.3f}+-{a.std():5.3f}"
    print(row)

print("\nInterpretacao: se FP ~ MV ~ IDW, os metodos sao equivalentes quando o no")
print("tem varios vizinhos-ancora; o FP nao demonstra superioridade de acuracia,")
print("mas integra-se ao mesmo grafo e opera com nos sem ancora imediata.")

import json
json.dump({m: {f: [float(x) for x in res[m][f]] for f in feat} for m in res},
          open(os.path.join(ART, "fp_mascaramento.json"), "w"))
