"""Estratificacao do INCREMENTO de chuva por regime (secao de resultados):
seco (< 0,2 mm), moderado (0,2-10 mm) e forte (>= 10 mm) na janela de 60 min.
Compara o MAE do incremento previsto pelo GraphSAGE com o da persistencia
(que preve incremento zero), nas 32 estacoes observadas.
"""
import os, pickle
import numpy as np

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]
obs = [i for i in range(len(nl)) if nl[i] not in set(g["blind_nodes"])]

X = np.asarray(d["X_2024"])
T = d["trues_chuva_all"].shape[0]
inc_real = (np.asarray(d["trues_chuva_all"]) - X[:T, :, 0])[:, obs]     # chuva que caiu em (t, t+6]
inc_prev = (np.asarray(d["preds_chuva_all"]) - X[:T, :, 0])[:, obs]     # incremento previsto pelo modelo
inc_pers = np.zeros_like(inc_real)                                       # persistencia: incremento zero

regimes = [("seco (<0,2 mm)", inc_real < 0.2),
           ("moderado (0,2-10 mm)", (inc_real >= 0.2) & (inc_real < 10)),
           ("forte (>=10 mm)", inc_real >= 10)]

print("MAE do INCREMENTO de chuva em 60 min por regime (32 estacoes obs, 2024):")
print(f"{'regime':>22} {'fracao':>8} {'chuva media':>12} {'GraphSAGE':>10} {'persist.':>9}")
for nome, m in regimes:
    fr = m.mean() * 100
    print(f"{nome:>22} {fr:7.1f}% {inc_real[m].mean():12.2f} "
          f"{np.abs(inc_prev[m] - inc_real[m]).mean():10.2f} "
          f"{np.abs(inc_pers[m] - inc_real[m]).mean():9.2f}")
