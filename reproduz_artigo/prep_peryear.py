"""Regenera tensores POR ANO (para o split de validação retida do item 3).

Reexecuta as celulas de dados do notebook (0..15: leitura dos CSVs + Feature
Propagation por ano) e salva X_treino[ano] separado + X_2024, para permitir
treinar em 2016-2022, validar em 2023 e testar em 2024 sem re-ler os CSVs.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import torch

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(CODIGO, "grafo-0.8.0-pyg.ipynb")
OUT = os.path.join(CODIGO, "artifacts", "peryear_cache.pt")
os.chdir(CODIGO)

nb = json.load(open(NB))
ns = {"__name__": "__main__"}
# Celulas 0..16: imports, grafo, CSVs, FP por ano (X_treino dict), X_2024.
for i in range(0, 17):
    c = nb["cells"][i]
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"]).replace("plt.show()", "pass")
    print(f"== celula {i} ==", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), ns)

X_treino = ns["X_treino"]                       # dict {ano: tensor [T,N,4]}
nodes_lista = [str(n) for n in ns["nodes_lista"]]
# idx dos nos com dados (exclui os 3 cegos), computado direto p/ nao depender da cel.17
idx_treino = torch.tensor([i for i, n in enumerate(nodes_lista)
                           if n not in {"33", "34", "35"}], dtype=torch.long)
torch.save({
    "X_treino": {int(a): X_treino[a].cpu() for a in X_treino},
    "X_2024":   ns["X_2024"].cpu(),
    "edge_index": ns["g_pyg"].edge_index.cpu(),
    "edge_weight": ns["g_pyg"].edge_weight.cpu(),
    "idx_treino": idx_treino,
    "nodes_lista": nodes_lista,
}, OUT)
print("\nCache por ano salvo:", OUT)
for a in sorted(X_treino):
    print(f"  {a}: {tuple(X_treino[a].shape)}")
print("  2024:", tuple(ns["X_2024"].shape))
