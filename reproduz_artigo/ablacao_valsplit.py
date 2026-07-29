"""Ablacao COM validacao retida (item 3 aplicado a ablacao): treino 2016-2022,
val 2023, teste 2024, early stopping monitorando a perda de VALIDACAO.
3 configs (MLP, SAGE sem pesos, SAGE ponderado) x 2 vars x 3 sementes.
Mesma logica de ablacao.py, mas com split e monitoramento de validacao.
"""
import os, sys, json
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(CODIGO, "artifacts")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
H, HF, LR, EP, BATCH = 6, 64, 1e-3, 300, 1024
PAT, MIND, LRPAT, LRFAC, MINLR = 25, 1e-5, 8, 0.5, 1e-5
ANOS_TR = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
ANO_VAL = 2023

c = torch.load(os.path.join(ART, "peryear_cache.pt"), weights_only=False)
Xty = {int(a): c["X_treino"][a] for a in c["X_treino"]}
X2024 = c["X_2024"]
ei = c["edge_index"].to(DEV)
w_geo = c["edge_weight"].to(DEV)
w_uni = torch.ones_like(w_geo)
idx = c["idx_treino"].to(DEV)
nl = c["nodes_lista"]
obs = [i for i, n in enumerate(nl) if n not in {"33", "34", "35"}]


def montar(anos, alvo):
    Xs, Ys = [], []
    for a in anos:
        Xa = Xty[a]; Xs.append(Xa[:-H])
        dif = Xa[H:, :, alvo] - Xa[:-H, :, alvo]
        Ys.append(dif.clamp(min=0) if alvo == 0 else dif)
    return torch.cat(Xs), torch.cat(Ys)


class SAGEConv(nn.Module):
    def __init__(s, ci, co):
        super().__init__(); s.a = nn.Linear(ci, co); s.b = nn.Linear(ci, co)

    def forward(s, x, e, ew):
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype).index_add_(0, e[1], ew).clamp(min=1e-8)
        agg = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, e[1], ew.view(-1, 1) * x[e[0]])
        return s.a(x) + s.b(agg / deg.unsqueeze(-1))


class GNN(nn.Module):
    def __init__(s):
        super().__init__(); s.c1 = SAGEConv(4, HF); s.c2 = SAGEConv(HF, 1)

    def forward(s, ew, xb):
        B, N, Fd = xb.shape; E = ei.size(1)
        offs = (torch.arange(B, device=xb.device) * N).repeat_interleave(E)
        eib = ei.repeat(1, B) + offs; ewb = ew.repeat(B)
        h = torch.relu(s.c1(xb.reshape(B * N, Fd), eib, ewb))
        return s.c2(h, eib, ewb).reshape(B, N)


class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = nn.Linear(4, HF); s.l2 = nn.Linear(HF, 1)

    def forward(s, ew, xb):
        B, N, Fd = xb.shape
        return s.l2(torch.relu(s.l1(xb.reshape(B * N, Fd)))).reshape(B, N)


@torch.no_grad()
def vloss(m, ew, Xv, Yv):
    m.eval(); tot = 0.0; nb = 0
    for a in range(0, len(Xv), BATCH):
        xb = Xv[a:a + BATCH].to(DEV); yb = Yv[a:a + BATCH].to(DEV)
        tot += float(((m(ew, xb)[:, idx] - yb[:, idx]) ** 2).mean().item()); nb += 1
    return tot / max(nb, 1)


def treinar(kind, ew, Xtr, Ytr, Xv, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = (MLP() if kind == "mlp" else GNN()).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=LRFAC, patience=LRPAT, min_lr=MINLR)
    crit = nn.MSELoss(); n = len(Xtr)
    best, noimp, bs = float("inf"), 0, None
    for ep in range(1, EP + 1):
        m.train(); perm = torch.randperm(n)
        for a in range(0, n, BATCH):
            sel = perm[a:a + BATCH]
            loss = crit(m(ew, Xtr[sel].to(DEV))[:, idx], Ytr[sel].to(DEV)[:, idx])
            opt.zero_grad(); loss.backward(); opt.step()
        vl = vloss(m, ew, Xv, Yv); sch.step(vl)
        if best - vl > MIND:
            best, noimp, bs = vl, 0, deepcopy(m.state_dict())
        else:
            noimp += 1
            if noimp >= PAT:
                break
    m.load_state_dict(bs); return m


@torch.no_grad()
def avaliar(m, ew, alvo, clip0):
    m.eval(); Xt = X2024[:-H]; outs = []
    for a in range(0, len(Xt), BATCH):
        outs.append(m(ew, Xt[a:a + BATCH].to(DEV)).cpu())
    res = torch.cat(outs)
    if clip0:
        res = res.clamp(min=0)
    pred = Xt[:, :, alvo] + res
    true = X2024[H:, :, alvo]
    e = (pred[:, obs] - true[:, obs]).numpy()
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


CONFIGS = [("MLP sem grafo", "mlp", w_uni),
           ("GraphSAGE sem pesos", "gnn", w_uni),
           ("GraphSAGE ponderado", "gnn", w_geo)]
res = {}
for nome, kind, ew in CONFIGS:
    for var, alvo, clip0 in [("PLU", 0, True), ("Temp", 1, False)]:
        Xtr, Ytr = montar(ANOS_TR, alvo); Xv, Yv = montar([ANO_VAL], alvo)
        vals = []
        for s in range(N_SEEDS):
            m = treinar(kind, ew, Xtr, Ytr, Xv, Yv, s)
            mae, rmse = avaliar(m, ew, alvo, clip0)
            print(f">> {nome} | {var} | s{s} -> MAE {mae:.4f} RMSE {rmse:.4f}", flush=True)
            vals.append((mae, rmse))
        res[(nome, var)] = np.array(vals)

print("\n" + "=" * 78)
print(f"ABLACAO com VALIDACAO RETIDA — teste 2024, agregado 32 obs ({N_SEEDS} sementes)")
print("=" * 78)
print(f"{'Config':22} {'PLU MAE':>14}{'PLU RMSE':>14}{'Temp MAE':>14}{'Temp RMSE':>14}")
for nome, _, _ in CONFIGS:
    p, t = res[(nome, "PLU")], res[(nome, "Temp")]
    print(f"{nome:22} {p[:,0].mean():7.3f}+-{p[:,0].std():.3f}{p[:,1].mean():7.3f}+-{p[:,1].std():.3f}"
          f"{t[:,0].mean():7.3f}+-{t[:,0].std():.3f}{t[:,1].mean():7.3f}+-{t[:,1].std():.3f}")
json.dump({f"{k[0]}|{k[1]}": v.tolist() for k, v in res.items()},
          open(os.path.join(ART, "ablacao_valsplit.json"), "w"), indent=2)
print("\nsalvo: artifacts/ablacao_valsplit.json")
