"""Ablacao pedida pelo prof. Ivan: isola o ganho do grafo e o da ponderacao.

Tres configuracoes, treinadas de forma IDENTICA (mesmos dados, mesmo alvo residual,
mesma loss mascarada, mesmo otimizador, mesmas epocas, mesma parada antecipada) e
variando UMA coisa de cada vez:

  MLP sem grafo   -> so a via "self" da camada: h = W_self x_i. Mede o ganho da
                     estrutura espacial (o modelo nao ve os vizinhos).
  SAGE sem pesos  -> WeightedSAGEConv com edge_weight = 1. Como a camada divide
                     pela soma dos pesos, isso e exatamente a media SIMPLES dos
                     vizinhos. Mede o ganho da ponderacao geografica.
  SAGE ponderado  -> modelo completo (pesos 1/Haversine).

O notebook NAO fixa semente, entao rodamos N_SEEDS sementes por configuracao e
reportamos media +- desvio: sem isso nao da para saber se a diferenca e real ou
ruido de inicializacao.

Uso:  .venv/bin/python codigo/reproduz_artigo/ablacao.py [n_seeds]
"""
import os, sys, pickle, json
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(CODIGO, "artifacts")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
HORIZON, H_FEATS, LR, MAX_EPOCHS, BATCH = 6, 64, 1e-3, 300, 1024
PATIENCE, MIN_DELTA, LR_PATIENCE, LR_FACTOR, MIN_LR = 25, 1e-5, 8, 0.5, 1e-5

cache = torch.load(os.path.join(ART, "train_cache.pt"), weights_only=False)
X_train = cache["X_train"]
Y_PLU, Y_TEMP = cache["Y_train"], cache["Y_train_temp"]
edge_index = cache["edge_index"].to(DEV)
w_geo = cache["edge_weight"].to(DEV)                 # D^-1 A dos pesos 1/Haversine
w_uni = torch.ones_like(w_geo)                       # sem ponderacao geografica
idx = cache["idx_treino"].to(DEV)
nodes_lista = cache["nodes_lista"]

test = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
X_2024 = torch.tensor(test["X_2024"])
X_test = X_2024[:-HORIZON]
Y_test_plu = X_2024[HORIZON:, :, 0]
Y_test_temp = X_2024[HORIZON:, :, 1]
obs = [i for i, n in enumerate(nodes_lista) if n not in {"33", "34", "35"}]


class SAGEConv(nn.Module):
    """Reproduz a WeightedSAGEConv do notebook: h = W_self x + W_neigh * (sum w x / sum w)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.lin_self = nn.Linear(cin, cout)
        self.lin_neigh = nn.Linear(cin, cout)

    def forward(self, x, ei, ew):
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, ei[1], ew)
        deg = deg.clamp(min=1e-8)
        agg = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, ei[1], ew.view(-1, 1) * x[ei[0]])
        return self.lin_self(x) + self.lin_neigh(agg / deg.unsqueeze(-1))


class GNN(nn.Module):
    def __init__(self, cin=4, h=H_FEATS):
        super().__init__()
        self.c1, self.c2 = SAGEConv(cin, h), SAGEConv(h, 1)

    def forward(self, ei, ew, xb):
        B, N, Fd = xb.shape
        E = ei.size(1)
        offs = (torch.arange(B, device=xb.device) * N).repeat_interleave(E)
        eib = ei.repeat(1, B) + offs
        ewb = ew.repeat(B)
        h = torch.relu(self.c1(xb.reshape(B * N, Fd), eib, ewb))
        return self.c2(h, eib, ewb).reshape(B, N)


class MLP(nn.Module):
    """Mesma profundidade/largura, sem o termo de vizinhanca: h = W_self x."""
    def __init__(self, cin=4, h=H_FEATS):
        super().__init__()
        self.l1, self.l2 = nn.Linear(cin, h), nn.Linear(h, 1)

    def forward(self, ei, ew, xb):
        B, N, Fd = xb.shape
        h = torch.relu(self.l1(xb.reshape(B * N, Fd)))
        return self.l2(h).reshape(B, N)


def treinar(kind, ew, Ytr, seed, nome):
    torch.manual_seed(seed); np.random.seed(seed)
    modelo = (MLP() if kind == "mlp" else GNN()).to(DEV)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=LR_FACTOR,
                                                       patience=LR_PATIENCE, min_lr=MIN_LR)
    crit = nn.MSELoss()
    n = len(X_train)
    best, no_imp, best_state = float("inf"), 0, None
    for ep in range(1, MAX_EPOCHS + 1):
        modelo.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for a in range(0, n, BATCH):
            sel = perm[a:a + BATCH]
            xb, yb = X_train[sel].to(DEV), Ytr[sel].to(DEV)
            loss = crit(modelo(edge_index, ew, xb)[:, idx], yb[:, idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        el = tot / max(nb, 1)
        sched.step(el)
        if best - el > MIN_DELTA:
            best, no_imp, best_state = el, 0, deepcopy(modelo.state_dict())
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
        if ep % 25 == 0:
            print(f"    [{nome}] epoca {ep:3d} | loss {el:.6f}", flush=True)
    if best_state: modelo.load_state_dict(best_state)
    print(f"    [{nome}] parou na epoca {ep} | melhor loss {best:.6f}", flush=True)
    return modelo


@torch.no_grad()
def avaliar(modelo, ew, alvo_idx, Yt, clip0):
    modelo.eval()
    outs = []
    for a in range(0, len(X_test), BATCH):
        xb = X_test[a:a + BATCH].to(DEV)
        r = modelo(edge_index, ew, xb).cpu()
        outs.append(r)
    res = torch.cat(outs)
    if clip0: res = res.clamp(min=0)
    pred = X_test[:, :, alvo_idx] + res           # x_hat = x(t) + delta
    e = (pred[:, obs] - Yt[:, obs]).numpy()       # agregado sobre os 32 observados
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


CONFIGS = [("MLP sem grafo", "mlp", w_uni),
           ("GraphSAGE sem pesos", "gnn", w_uni),
           ("GraphSAGE ponderado", "gnn", w_geo)]

res = {}
for nome, kind, ew in CONFIGS:
    for var, Ytr, ai, Yt, clip0 in [("PLU", Y_PLU, 0, Y_test_plu, True),
                                    ("Temp", Y_TEMP, 1, Y_test_temp, False)]:
        vals = []
        for s in range(N_SEEDS):
            print(f"\n>> {nome} | {var} | semente {s}", flush=True)
            m = treinar(kind, ew, Ytr, s, f"{nome}/{var}/s{s}")
            mae, rmse = avaliar(m, ew, ai, Yt, clip0)
            print(f"    -> MAE {mae:.4f} | RMSE {rmse:.4f}", flush=True)
            vals.append((mae, rmse))
        res[(nome, var)] = np.array(vals)

print("\n" + "=" * 78)
print(f"ABLACAO — teste 2024, agregado sobre os 32 nos observados ({N_SEEDS} sementes)")
print("=" * 78)
print(f"{'Configuracao':22} {'PLU MAE':>14} {'PLU RMSE':>14} {'Temp MAE':>14} {'Temp RMSE':>14}")
for nome, _, _ in CONFIGS:
    p, t = res[(nome, "PLU")], res[(nome, "Temp")]
    print(f"{nome:22} {p[:,0].mean():7.3f}±{p[:,0].std():.3f} {p[:,1].mean():7.3f}±{p[:,1].std():.3f} "
          f"{t[:,0].mean():7.3f}±{t[:,0].std():.3f} {t[:,1].mean():7.3f}±{t[:,1].std():.3f}")
print("\nreferencia do artigo (003): Persistencia 0.359/2.856 0.861/1.400 | "
      "GraphSAGE 0.496/2.885 0.879/1.335")

json.dump({f"{k[0]}|{k[1]}": v.tolist() for k, v in res.items()},
          open(os.path.join(ART, "ablacao.json"), "w"), indent=2)
print(f"\nresultados brutos salvos em {os.path.join(ART, 'ablacao.json')}")
