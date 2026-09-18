"""Custo computacional dos modelos: parametros, tamanho em disco, operacoes por
inferencia e latencia medida.

Converte em numero a afirmacao de custo baixo repetida no relatorio (Secoes 4,
5.4 e 9). Nao depende de retreinar: o GraphSAGE le os pesos versionados em
`artifacts/model_plu.pt`; o STGCN e instanciado com a arquitetura de
`stgcn_valsplit.py` (mesma contagem de parametros do modelo treinado).

Latencia = media de N inferencias sobre um instante da rede completa (35 nos),
em CPU, apos aquecimento. Reportar o equipamento junto do numero.

Uso: python reproduz_artigo/custo_computacional.py [n_repeticoes]
"""
import os
import pickle
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regen_x_test import MeteorologiaGNN  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(RAIZ, "artifacts")
DEV = torch.device("cpu")
N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 100
HF, W, KT, H = 64, 12, 3, 6

g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]
N = len(nl)
ei = torch.tensor(g["edge_index"], dtype=torch.long)
ew = torch.tensor(g["edge_weight"], dtype=torch.float32)
E = ei.size(1)
F_IN = len(g["feature_names"])

# ---------------------------------------------------------------- STGCN
A = torch.zeros(N, N)
A[ei[1], ei[0]] = ew
A = A / A.sum(dim=1, keepdim=True).clamp(min=1e-8)
At = A.t()


class TConv(nn.Module):
    def __init__(s, ci, co):
        super().__init__(); s.conv = nn.Conv2d(ci, 2 * co, (KT, 1))

    def forward(s, x):
        p, q = s.conv(x).chunk(2, dim=1)
        return p * torch.sigmoid(q)


class SConv(nn.Module):
    def __init__(s, ci, co):
        super().__init__(); s.a = nn.Conv2d(ci, co, 1); s.b = nn.Conv2d(ci, co, 1)

    def forward(s, x):
        return s.a(x) + s.b(x.matmul(At))


class Bloco(nn.Module):
    def __init__(s, ci, co):
        super().__init__()
        s.t1 = TConv(ci, co); s.g = SConv(co, co); s.t2 = TConv(co, co)
        s.ln = nn.LayerNorm(co)

    def forward(s, x):
        x = s.t2(torch.relu(s.g(s.t1(x))))
        return s.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class STGCN(nn.Module):
    def __init__(s):
        super().__init__()
        s.b1 = Bloco(F_IN, HF); s.b2 = Bloco(HF, HF)
        t_rest = W - 4 * (KT - 1)
        s.tf = TConv(HF, HF); s.tf.conv = nn.Conv2d(HF, 2 * HF, (t_rest, 1))
        s.o1 = nn.Conv2d(HF, HF, 1); s.o2 = nn.Conv2d(HF, 1, 1)

    def forward(s, xw):
        return s.o2(torch.relu(s.o1(s.tf(s.b2(s.b1(xw)))))).squeeze(1).squeeze(1)


def latencia(fn, x, n=N_REP):
    with torch.no_grad():
        for _ in range(10):
            fn(x)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(x); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3
    return ts.mean(), ts.std()


print("=" * 68)
print(f"Equipamento: {platform.processor() or platform.machine()} | "
      f"{platform.system()} {platform.release()} | torch {torch.__version__} | CPU")
print(f"Grafo: N={N} nos, E={E} arestas direcionadas, F={F_IN} features")
print(f"Repeticoes por medicao: {N_REP}")
print("=" * 68)

sage = MeteorologiaGNN(F_IN, HF).to(DEV).eval()
sage.load_state_dict(torch.load(os.path.join(ART, "model_plu.pt"), map_location=DEV))
n_sage = sum(p.numel() for p in sage.parameters())
x1 = torch.randn(1, N, F_IN)
m_sage, s_sage = latencia(lambda x: sage.forward_batched(ei, ew, x), x1)

stgcn = STGCN().to(DEV).eval()
n_stgcn = sum(p.numel() for p in stgcn.parameters())
xw = torch.randn(1, F_IN, W, N)
m_st, s_st = latencia(stgcn, xw)

# operacoes de multiplicacao-acumulacao por inferencia (1 instante, rede toda)
# WeightedSAGEConv(ci->co): N*ci*co (self) + E*ci (agregacao) + N*ci*co (neigh)
mac_sage = (2 * N * F_IN * HF + E * F_IN) + (2 * N * HF * 1 + E * HF)
kb = lambda p: os.path.getsize(p) / 1024

print(f"\nGraphSAGE (por variavel-alvo)")
print(f"  parametros            : {n_sage}")
print(f"  arquivo de pesos      : {kb(os.path.join(ART, 'model_plu.pt')):.1f} KB")
print(f"  mult-acum / inferencia: {mac_sage:,}")
print(f"  latencia (1 instante) : {m_sage:.3f} +- {s_sage:.3f} ms")
print(f"  sistema completo (PLU + Temp): {2 * n_sage} parametros, "
      f"{2 * m_sage:.3f} ms")

print(f"\nSTGCN (janela W={W})")
print(f"  parametros            : {n_stgcn}")
print(f"  latencia (1 instante) : {m_st:.3f} +- {s_st:.3f} ms")
print(f"  razao de parametros STGCN/GraphSAGE: {n_stgcn / n_sage:.0f}x")
print(f"  razao de latencia     : {m_st / m_sage:.1f}x")

# throughput: quantos instantes da rede por segundo
print(f"\nVazao GraphSAGE: {1000 / m_sage:,.0f} instantes/s "
      f"(a rede produz 1 instante a cada 10 min)")
