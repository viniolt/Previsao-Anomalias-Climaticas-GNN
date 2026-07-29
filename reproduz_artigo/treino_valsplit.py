"""Item 3 do prof.: validacao retida. Treina em 2016-2022, valida em 2023
(early stopping + reducao de lr monitorando a perda de VALIDACAO, nao de treino),
testa em 2024. Produz um artefato no formato de X_test.pkl para os scripts de
analise (numeros.py, deteccao_reforcada.py, figuras) regenerarem os numeros.

Climatologia (tau, sigma_inv) permanece sobre 2016-2023 (estatistica, sem
vazamento para 2024), entao Tabela IV e limiares de deteccao nao dependem disto.
"""
import os, pickle
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(CODIGO, "artifacts")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, HF, LR, EP, BATCH = 6, 64, 1e-3, 300, 1024
PAT, MIND, LRPAT, LRFAC, MINLR = 25, 1e-5, 8, 0.5, 1e-5
SEED = 0

c = torch.load(os.path.join(ART, "peryear_cache.pt"), weights_only=False)
Xty = {int(a): c["X_treino"][a] for a in c["X_treino"]}
X2024 = c["X_2024"]
ei = c["edge_index"].to(DEV); ew = c["edge_weight"].to(DEV)
idx = c["idx_treino"].to(DEV); nl = c["nodes_lista"]

ANOS_TR = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
ANO_VAL = 2023


def montar(anos, alvo):
    Xs, Ys = [], []
    for a in anos:
        Xa = Xty[a]
        Xs.append(Xa[:-H])
        dif = Xa[H:, :, alvo] - Xa[:-H, :, alvo]
        Ys.append(dif.clamp(min=0) if alvo == 0 else dif)
    return torch.cat(Xs), torch.cat(Ys)


class SAGEConv(nn.Module):
    def __init__(s, ci, co):
        super().__init__(); s.a = nn.Linear(ci, co); s.b = nn.Linear(ci, co)

    def forward(s, x, ei, ew):
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype).index_add_(0, ei[1], ew).clamp(min=1e-8)
        agg = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, ei[1], ew.view(-1, 1) * x[ei[0]])
        return s.a(x) + s.b(agg / deg.unsqueeze(-1))


class GNN(nn.Module):
    def __init__(s):
        super().__init__(); s.c1 = SAGEConv(4, HF); s.c2 = SAGEConv(HF, 1)

    def forward(s, xb):
        B, N, Fd = xb.shape; E = ei.size(1)
        offs = (torch.arange(B, device=xb.device) * N).repeat_interleave(E)
        eib = ei.repeat(1, B) + offs; ewb = ew.repeat(B)
        h = torch.relu(s.c1(xb.reshape(B * N, Fd), eib, ewb))
        return s.c2(h, eib, ewb).reshape(B, N)


@torch.no_grad()
def val_loss(m, Xv, Yv):
    m.eval(); tot = 0.0; nb = 0
    for a in range(0, len(Xv), BATCH):
        xb = Xv[a:a + BATCH].to(DEV); yb = Yv[a:a + BATCH].to(DEV)
        tot += float(((m(xb)[:, idx] - yb[:, idx]) ** 2).mean().item()); nb += 1
    return tot / max(nb, 1)


def treinar(Xtr, Ytr, Xv, Yv, nome):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = GNN().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=LRFAC, patience=LRPAT, min_lr=MINLR)
    crit = nn.MSELoss(); n = len(Xtr)
    best, noimp, best_state = float("inf"), 0, None
    for ep in range(1, EP + 1):
        m.train(); perm = torch.randperm(n)
        for a in range(0, n, BATCH):
            sel = perm[a:a + BATCH]
            xb = Xtr[sel].to(DEV); yb = Ytr[sel].to(DEV)
            loss = crit(m(xb)[:, idx], yb[:, idx])
            opt.zero_grad(); loss.backward(); opt.step()
        vl = val_loss(m, Xv, Yv)                 # <-- monitora VALIDACAO (2023)
        sch.step(vl)
        if best - vl > MIND:
            best, noimp, best_state = vl, 0, deepcopy(m.state_dict())
        else:
            noimp += 1
            if noimp >= PAT:
                break
        if ep % 20 == 0:
            print(f"  [{nome}] epoca {ep:3d} | val_loss {vl:.6f}", flush=True)
    m.load_state_dict(best_state)
    print(f"  [{nome}] parou epoca {ep} | melhor val_loss {best:.6f}", flush=True)
    return m


@torch.no_grad()
def prever(m, alvo, clip0):
    m.eval(); Xt = X2024[:-H]; outs = []
    for a in range(0, len(Xt), BATCH):
        outs.append(m(Xt[a:a + BATCH].to(DEV)).cpu())
    res = torch.cat(outs)
    if clip0:
        res = res.clamp(min=0)
    return (Xt[:, :, alvo] + res).numpy()        # x_hat absoluto [T,N]


print("=== ITEM 3: validacao retida (treino 2016-2022, val 2023, teste 2024) ===")
Xtr_p, Ytr_p = montar(ANOS_TR, 0); Xv_p, Yv_p = montar([ANO_VAL], 0)
Xtr_t, Ytr_t = montar(ANOS_TR, 1); Xv_t, Yv_t = montar([ANO_VAL], 1)
print(f"treino {tuple(Xtr_p.shape)} | val {tuple(Xv_p.shape)} | 2016-2022 + 2023", flush=True)

mp = treinar(Xtr_p, Ytr_p, Xv_p, Yv_p, "PLU")
mt = treinar(Xtr_t, Ytr_t, Xv_t, Yv_t, "Temp")

preds_chuva = prever(mp, 0, True)                # [T,N]
preds_temp = prever(mt, 1, False)

# reaproveita o X_test.pkl atual (trues, sigma, timestamps) e troca so as previsoes
old = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
new = dict(old)
new["preds_chuva_all"] = preds_chuva.astype(np.float32)
new["preds_temp_all"] = preds_temp.astype(np.float32)
# recomputa anomaly_scores com as novas previsoes (PLU,Temp do modelo; UR,vento persistencia)
sig = pickle.load(open(os.path.join(ART, "sigma_inv.pkl"), "rb"))   # [N,4,4] sobre 2016-2023
T = preds_chuva.shape[0]
xhat = X2024[:T].numpy().copy()                  # persistencia p/ todas
xhat[:, :, 0] = preds_chuva
xhat[:, :, 1] = preds_temp
e = old["X_real_next"] - xhat                    # erro [T,N,4]
AS = np.zeros((T, 35))
for i in range(35):
    ei_ = e[:, i, :]
    AS[:, i] = np.sqrt(np.einsum("tf,fg,tg->t", ei_, sig[i], ei_).clip(min=0))
new["anomaly_scores"] = AS
pickle.dump(new, open(os.path.join(ART, "X_test_valsplit.pkl"), "wb"))
print("\nArtefato salvo: artifacts/X_test_valsplit.pkl (previsoes + anomaly_scores novos)")
