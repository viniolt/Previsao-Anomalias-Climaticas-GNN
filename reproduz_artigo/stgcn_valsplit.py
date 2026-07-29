"""STGCN (Yu et al., IJCAI 2018, adaptado) vs GraphSAGE — mesmo protocolo da
ablacao com validacao retida: treino 2016-2022, val 2023, teste 2024, alvo
residual em t+H (H=6 passos = 60 min), perda/avaliacao nas 32 estacoes obs,
early stopping na perda de VALIDACAO, ReduceLROnPlateau, 3 sementes x 2 vars.

Diferenca central p/ o GraphSAGE do artigo: o STGCN ve uma JANELA de W=12
passos passados (2 h) por no, nao so o instante t. Arquitetura: 2 blocos
ST (T-conv GLU -> conv espacial D^-1A -> T-conv GLU) + colapso temporal final.
A conv espacial usa a mesma agregacao do artigo (termo proprio + media
ponderada dos vizinhos com pesos geograficos row-normalizados).

Avaliacao nas MESMAS ancoras t>=W-1 para todos; imprime persistencia no mesmo
conjunto de ancoras para leitura direta.

Uso: stgcn_valsplit.py [n_sementes] [--smoke]
"""
import os, sys, json, time
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(CODIGO, "artifacts")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = "--smoke" in sys.argv
argv = [a for a in sys.argv[1:] if a != "--smoke"]
N_SEEDS = int(argv[0]) if argv else 3
H, W, KT, HF, LR, EP, BATCH = 6, 12, 3, 64, 1e-3, 300, 1024
PAT, MIND, LRPAT, LRFAC, MINLR = 25, 1e-5, 8, 0.5, 1e-5
ANOS_TR = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
ANO_VAL = 2023
if SMOKE:
    EP, N_SEEDS = 2, 1

c = torch.load(os.path.join(ART, "peryear_cache.pt"), weights_only=False)
ei = c["edge_index"]; w_geo = c["edge_weight"]
idx = c["idx_treino"].to(DEV)
nl = c["nodes_lista"]; N = len(nl)
obs = [i for i, n in enumerate(nl) if n not in {"33", "34", "35"}]

# adjacencia densa row-estocastica (D^-1 A) so de vizinhos; termo proprio a parte
A = torch.zeros(N, N)
A[ei[1], ei[0]] = w_geo
A = A / A.sum(dim=1, keepdim=True).clamp(min=1e-8)
At = A.t().to(DEV)                                     # p/ matmul no ultimo eixo


def concat_anos(anos):
    """Concatena anos e devolve (Xcat, ancoras t validas sem cruzar fronteira)."""
    Xs, anc, off = [], [], 0
    for a in anos:
        Xa = c["X_treino"][a] if a != 2024 else c["X_2024"]
        Ta = Xa.shape[0]
        anc.append(torch.arange(off + W - 1, off + Ta - H))
        Xs.append(Xa); off += Ta
    return torch.cat(Xs).to(DEV), torch.cat(anc).to(DEV)


OFFS = torch.arange(-W + 1, 1, device=DEV)             # janela [t-W+1 .. t]


def lote(Xcat, ts, alvo):
    """[B,F,W,N] janela + alvo residual em t+H (clamp>=0 p/ chuva)."""
    xw = Xcat[ts.unsqueeze(1) + OFFS]                  # [B,W,N,F]
    dif = Xcat[ts + H, :, alvo] - Xcat[ts, :, alvo]
    y = dif.clamp(min=0) if alvo == 0 else dif
    return xw.permute(0, 3, 1, 2).contiguous(), y


class TConv(nn.Module):
    """Convolucao temporal com GLU (kernel KT no eixo do tempo)."""
    def __init__(s, ci, co):
        super().__init__(); s.conv = nn.Conv2d(ci, 2 * co, (KT, 1))

    def forward(s, x):                                 # x [B,C,T,N]
        p, q = s.conv(x).chunk(2, dim=1)
        return p * torch.sigmoid(q)


class SConv(nn.Module):
    """Conv espacial: termo proprio + media ponderada dos vizinhos (D^-1 A)."""
    def __init__(s, ci, co):
        super().__init__()
        s.a = nn.Conv2d(ci, co, 1); s.b = nn.Conv2d(ci, co, 1)

    def forward(s, x):                                 # x [B,C,T,N]
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
        s.b1 = Bloco(4, HF); s.b2 = Bloco(HF, HF)
        t_rest = W - 4 * (KT - 1)                      # 12 -> 4
        s.tf = TConv(HF, HF); s.tf.conv = nn.Conv2d(HF, 2 * HF, (t_rest, 1))
        s.o1 = nn.Conv2d(HF, HF, 1); s.o2 = nn.Conv2d(HF, 1, 1)

    def forward(s, xw):                                # [B,F,W,N] -> [B,N]
        h = s.tf(s.b2(s.b1(xw)))                       # [B,HF,1,N]
        return s.o2(torch.relu(s.o1(h))).squeeze(1).squeeze(1)


@torch.no_grad()
def vloss(m, Xv, av, alvo):
    m.eval(); tot, nb = 0.0, 0
    for a in range(0, len(av), BATCH):
        xb, yb = lote(Xv, av[a:a + BATCH], alvo)
        tot += float(((m(xb)[:, idx] - yb[:, idx]) ** 2).mean().item()); nb += 1
    return tot / max(nb, 1)


def treinar(Xtr, atr, Xv, av, alvo, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = STGCN().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=LRFAC, patience=LRPAT, min_lr=MINLR)
    crit = nn.MSELoss(); n = len(atr)
    best, noimp, bs = float("inf"), 0, None
    for ep in range(1, EP + 1):
        m.train(); perm = torch.randperm(n, device=DEV)
        if SMOKE:
            perm = perm[:BATCH * 10]
        for a in range(0, len(perm), BATCH):
            xb, yb = lote(Xtr, atr[perm[a:a + BATCH]], alvo)
            loss = crit(m(xb)[:, idx], yb[:, idx])
            opt.zero_grad(); loss.backward(); opt.step()
        vl = vloss(m, Xv, av, alvo); sch.step(vl)
        if best - vl > MIND:
            best, noimp, bs = vl, 0, deepcopy(m.state_dict())
        else:
            noimp += 1
            if noimp >= PAT:
                break
        if ep % 10 == 0 or ep == 1:
            print(f"    ep {ep:3d}  val {vl:.5f}  best {best:.5f}", flush=True)
    m.load_state_dict(bs); return m, ep, best


@torch.no_grad()
def avaliar(m, Xte, ate, alvo, clip0):
    m.eval(); outs = []
    for a in range(0, len(ate), BATCH):
        xb, _ = lote(Xte, ate[a:a + BATCH], alvo)
        outs.append(m(xb).cpu())
    res = torch.cat(outs)
    if clip0:
        res = res.clamp(min=0)
    tcpu = ate.cpu(); Xc = Xte.cpu()
    pred = Xc[tcpu, :, alvo] + res
    true = Xc[tcpu + H, :, alvo]
    e = (pred[:, obs] - true[:, obs]).numpy()
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


Xtr, atr = concat_anos(ANOS_TR)
Xv, av = concat_anos([ANO_VAL])
Xte, ate = concat_anos([2024])
print(f"treino: {len(atr)} janelas | val: {len(av)} | teste: {len(ate)} | dev {DEV}", flush=True)

# persistencia nas MESMAS ancoras do teste (referencia exata)
tcpu = ate.cpu(); Xc = Xte.cpu()
for var, alvo in [("PLU", 0), ("Temp", 1)]:
    e = (Xc[tcpu, :, alvo][:, obs] - Xc[tcpu + H, :, alvo][:, obs]).numpy()
    print(f"persistencia {var}: MAE {np.abs(e).mean():.3f} RMSE {np.sqrt((e**2).mean()):.3f}")

res = {}
for var, alvo, clip0 in [("PLU", 0, True), ("Temp", 1, False)]:
    vals = []
    for s in range(N_SEEDS):
        t0 = time.time()
        m, eps, bv = treinar(Xtr, atr, Xv, av, alvo, s)
        mae, rmse = avaliar(m, Xte, ate, alvo, clip0)
        print(f">> STGCN | {var} | s{s} -> MAE {mae:.4f} RMSE {rmse:.4f} "
              f"({eps} eps, val {bv:.5f}, {time.time()-t0:.0f}s)", flush=True)
        vals.append((mae, rmse))
    res[var] = np.array(vals)

print("\n" + "=" * 70)
print(f"STGCN com VALIDACAO RETIDA — teste 2024, agregado 32 obs ({N_SEEDS} sementes)")
print("=" * 70)
for var in ["PLU", "Temp"]:
    v = res[var]
    print(f"{var:5} MAE {v[:,0].mean():.3f}+-{v[:,0].std():.3f}  RMSE {v[:,1].mean():.3f}+-{v[:,1].std():.3f}")
if not SMOKE:
    json.dump({k: v.tolist() for k, v in res.items()},
              open(os.path.join(ART, "stgcn_valsplit.json"), "w"), indent=2)
    print("\nsalvo: artifacts/stgcn_valsplit.json")
