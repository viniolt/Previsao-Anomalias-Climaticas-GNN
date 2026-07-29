"""Antes/depois da validacao retida (item 3): regenera Tabela III (previsao),
correlacao de temperatura e Tabela V (deteccao) a partir do artefato novo
X_test_valsplit.pkl e compara com o modelo atual (X_test.pkl).
"""
import pickle, os
import numpy as np
import pandas as pd

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]; blind = set(g["blind_nodes"])
obs = [i for i in range(35) if nl[i] not in blind]


def agg(pred, true):
    e = pred - true
    return np.abs(e).mean(), np.sqrt((e ** 2).mean())


def tabela(path, nome):
    d = pickle.load(open(path, "rb"))
    X = d["X_2024"]; T = d["trues_chuva_all"].shape[0]
    ts = pd.to_datetime(d["timestamps_2024"])
    r = {}
    for var, tk, pk, idx in [("PLU", "trues_chuva_all", "preds_chuva_all", 0),
                             ("Temp", "trues_temp_all", "preds_temp_all", 1)]:
        t = d[tk][:, obs]; p = d[pk][:, obs]
        r[("GraphSAGE", var)] = agg(p, t)
        r[("Persist", var)] = agg(X[:T][:, obs, idx], t)
    tt = d["trues_temp_all"][:, obs].ravel(); pp = d["preds_temp_all"][:, obs].ravel()
    ok = np.isfinite(tt) & np.isfinite(pp)
    corr = np.corrcoef(tt[ok], pp[ok])[0, 1]

    # deteccao
    inc = d["trues_chuva_all"] - X[:T, :, 0]
    AS = d["anomaly_scores"]
    dias = pd.Series(pd.DatetimeIndex(ts[6:T + 6]).date)
    ev = pd.DataFrame({"d": dias.values, "m": inc[:, obs].max(axis=1)}).groupby("d")["m"].max()
    evento = (ev > 10).values
    S = AS[:, obs]
    smax = pd.DataFrame({"d": dias.values, "m": S.max(axis=1)}).groupby("d")["m"].max().reindex(ev.index).values
    det = {}
    for pct in (99, 99.9):
        thr = np.nanpercentile(S, pct); flag = smax > thr
        tp = int((flag & evento).sum()); fp = int((flag & ~evento).sum()); fn = int((~flag & evento).sum())
        det[pct] = (tp / (tp + fn) if tp + fn else 0, tp / (tp + fp) if tp + fp else 0)

    print(f"\n===== {nome} =====")
    print(f"{'':14}{'PLU MAE':>9}{'PLU RMSE':>9}{'Temp MAE':>9}{'Temp RMSE':>10}")
    for m in ["Persist", "GraphSAGE"]:
        a = r[(m, "PLU")]; b = r[(m, "Temp")]
        print(f"{m:14}{a[0]:9.3f}{a[1]:9.3f}{b[0]:9.3f}{b[1]:10.3f}")
    print(f"corr Temp: {corr:.4f}")
    print(f"deteccao p99: recall {det[99][0]*100:.0f}% prec {det[99][1]*100:.1f}% | "
          f"p99.9: recall {det[99.9][0]*100:.0f}% prec {det[99.9][1]*100:.1f}%")
    return r, corr, det


tabela(os.path.join(ART, "X_test.pkl"), "ATUAL (treino 2016-2023, early stop na perda de treino)")
vs = os.path.join(ART, "X_test_valsplit.pkl")
if os.path.exists(vs):
    tabela(vs, "VALIDACAO RETIDA (treino 2016-2022, val 2023)")
else:
    print("\n(X_test_valsplit.pkl ainda nao existe)")
