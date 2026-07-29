"""Metricas de previsao POR ESTACAO (secao de resultados): MAE/RMSE do
acumulado de chuva e da temperatura em t+6, para cada uma das 32 estacoes
observadas, com minimo, mediana e maximo do RMSE.
"""
import os, pickle
import numpy as np

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]; lab = g["labels"]
obs = [i for i in range(len(nl)) if nl[i] not in set(g["blind_nodes"])]

res = {}
for var, pred, true in [("PLU", "preds_chuva_all", "trues_chuva_all"),
                        ("Temp", "preds_temp_all", "trues_temp_all")]:
    e = np.asarray(d[pred]) - np.asarray(d[true])
    res[var] = {i: (np.abs(e[:, i]).mean(), np.sqrt((e[:, i] ** 2).mean())) for i in obs}

print(f"{'no':>3} {'estacao':>32} {'PLU MAE':>8} {'PLU RMSE':>9} {'Temp MAE':>9} {'Temp RMSE':>10}")
for i in obs:
    p, t = res["PLU"][i], res["Temp"][i]
    print(f"{nl[i]:>3} {lab.get(nl[i], '').strip()[:32]:>32} {p[0]:8.2f} {p[1]:9.2f} {t[0]:9.2f} {t[1]:10.2f}")

for var in ["Temp", "PLU"]:
    r = np.array([res[var][i][1] for i in obs])
    print(f"\n{var}: RMSE por estacao min {r.min():.2f} | mediana {np.median(r):.2f} | max {r.max():.2f}")
