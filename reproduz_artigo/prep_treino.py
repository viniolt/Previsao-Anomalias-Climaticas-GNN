"""Reconstroi X_train/Y_train reexecutando as celulas de dados do proprio notebook.

Reexecutar o notebook (em vez de reimplementar o pipeline) garante que os tensores
de treino sao identicos aos que produziram os numeros do artigo. Salva um cache em
artifacts/train_cache.pt para as ablacoes nao precisarem refazer a leitura dos CSVs.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import torch

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(CODIGO, "grafo-0.8.0-pyg.ipynb")
OUT = os.path.join(CODIGO, "artifacts", "train_cache.pt")

os.chdir(CODIGO)  # o notebook le "003.graphml" por caminho relativo

nb = json.load(open(NB))
ns = {"__name__": "__main__"}

# Celulas 0..17: imports, grafo, coordenadas, leitura dos CSVs, FP e montagem
# de X_train/Y_train. Paramos em 17 - dali em diante e treino/plot.
for i in range(0, 18):
    c = nb["cells"][i]
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if "plt.show()" in src:
        src = src.replace("plt.show()", "pass")
    print(f"\n{'='*20} executando celula {i} {'='*20}", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), ns)

torch.save({
    "X_train":      ns["X_train"].cpu(),
    "Y_train":      ns["Y_train"].cpu(),
    "Y_train_temp": ns["Y_train_temp"].cpu(),
    "edge_index":   ns["g_pyg"].edge_index.cpu(),
    "edge_weight":  ns["g_pyg"].edge_weight.cpu(),
    "idx_treino":   ns["IDX_TREINO"].cpu(),
    "nodes_lista":  [str(n) for n in ns["nodes_lista"]],
}, OUT)

print(f"\nCache salvo em {OUT}")
print("  X_train:", tuple(ns["X_train"].shape), "| Y_train:", tuple(ns["Y_train"].shape))
print("  arestas dirigidas:", ns["g_pyg"].edge_index.size(1))
