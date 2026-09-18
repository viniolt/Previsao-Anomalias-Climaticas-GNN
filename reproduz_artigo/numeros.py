import os, pickle, numpy as np, pandas as pd
from caminhos import ARTIFACTS

d = pickle.load(open(os.path.join(ARTIFACTS, 'X_test.pkl'), 'rb'))
g = pickle.load(open(os.path.join(ARTIFACTS, 'graph_data.pkl'), 'rb'))
lab = g['labels']; nl = g['nodes_lista']
blind = [nl.index(b) for b in g['blind_nodes']]
obs = [i for i in range(35) if i not in blind]
X = d['X_2024']; T = d['trues_chuva_all'].shape[0]
ts = pd.to_datetime(d['timestamps_2024'])


def agg(pred, true):
    e = pred - true
    return np.abs(e).mean(), np.sqrt((e ** 2).mean())


print("=" * 72)
print("TABELA III  (agregado sobre os 32 nos observados - convencao do artigo)")
print("=" * 72)
rows = {}
for nome, idx, tk, pk in [('PLU', 0, 'trues_chuva_all', 'preds_chuva_all'),
                          ('Temp', 1, 'trues_temp_all', 'preds_temp_all')]:
    t = d[tk][:, obs]; p = d[pk][:, obs]
    rows[('GraphSAGE', nome)] = agg(p, t)
    rows[('Persistencia', nome)] = agg(X[:T][:, obs, idx], t)

print(f"{'Modelo':14s} {'PLU MAE':>9} {'PLU RMSE':>9} {'Temp MAE':>9} {'Temp RMSE':>10}")
print(f"{'Media hist':14s} {'2.917':>9} {'6.767':>9} {'3.634':>9} {'4.631':>10}   <- nao usa grafo, inalterada")
for m in ['Persistencia', 'GraphSAGE']:
    a = rows[(m, 'PLU')]; b = rows[(m, 'Temp')]
    print(f"{m:14s} {a[0]:9.3f} {a[1]:9.3f} {b[0]:9.3f} {b[1]:10.3f}")
print()
print("artigo (valsplit): Persist 0,359/2,856 0,861/1,400 | modelo implantado 0,535/2,888 0,874/1,332")

# correlacao da temperatura
tt = d['trues_temp_all'][:, obs].ravel(); pp = d['preds_temp_all'][:, obs].ravel()
ok = np.isfinite(tt) & np.isfinite(pp)
print(f"\ncorrelacao Temp (previsto x observado): {np.corrcoef(tt[ok], pp[ok])[0,1]:.4f}   (artigo: 0,96)")

# ---------------- Tabela V ----------------
inc = d['trues_chuva_all'] - X[:T, :, 0]
AS = d['anomaly_scores']
dias = pd.Series(pd.DatetimeIndex(ts[6:T + 6]).date)
ev = pd.DataFrame({'d': dias.values, 'm': inc[:, obs].max(axis=1)}).groupby('d')['m'].max()
evento = ev > 10
S = AS[:, obs]
smax = pd.DataFrame({'d': dias.values, 'm': S.max(axis=1)}).groupby('d')['m'].max().reindex(evento.index)

print("\n" + "=" * 72)
print(f"TABELA V   (evento = algum posto > 10mm/60min no dia: {int(evento.sum())} de {len(evento)} dias)")
print("=" * 72)
for p in (99, 99.9):
    thr = np.nanpercentile(S, p)
    flag = smax > thr
    tp = int((flag & evento).sum()); fp = int((flag & ~evento).sum()); fn = int((~flag & evento).sum())
    print(f"  Percentil {p:<5}: recall = {tp/(tp+fn)*100:5.1f} %   precisao = {tp/(tp+fp)*100:5.1f} %")
print("artigo (valsplit): p99 -> 100% / 26,6%  |  p99.9 -> 82,5% / 43,7%")

# ---------------- Estudo de caso 13/01 ----------------
print("\n" + "=" * 72)
print("ESTUDO DE CASO 13/01/2024")
print("=" * 72)
tsr = pd.DatetimeIndex(ts[6:T + 6])
thr99 = np.nanpercentile(S, 99)
alvo = (tsr.date == pd.Timestamp('2024-01-13').date()) & (tsr.hour == 0) & (tsr.minute == 50)
k = int(np.where(alvo)[0][0])
n_acima = int((AS[k, obs] > thr99).sum())
print(f"  as 00h50: {n_acima} de 32 estacoes com score > p99   (artigo: 25 de 32)")

dia = tsr.date == pd.Timestamp('2024-01-13').date()
smax_dia = AS[dia][:, obs].max()
no_max = obs[int(np.unravel_index(AS[dia][:, obs].argmax(), AS[dia][:, obs].shape)[1])]
print(f"  maior score do dia: {smax_dia:.1f} no no {nl[no_max]} ({lab[nl[no_max]]})   (artigo: 43,0 em Capela do Socorro - Subprefeitura)")

inc_dia = inc[dia][:, obs]
mx = float(inc_dia.max())
no_chuva = obs[int(np.unravel_index(inc_dia.argmax(), inc_dia.shape)[1])]
print(f"  maior chuva real em 60 min: {mx:.1f} mm no no {nl[no_chuva]} ({lab[nl[no_chuva]]})   (artigo: 61,5 mm em Barragem Palheiros)")

# FP do no 33 no dia 13/01 (total diario) - deve bater com 82,6 mm do artigo
d13 = pd.DatetimeIndex(ts[:len(X)]).date == pd.Timestamp('2024-01-13').date()
plu33 = X[d13][:, nl.index('33'), 0]
print(f"  FP no no 33 em 13/01: acumulado do dia = {float(plu33.max() - plu33.min()):.1f} mm   (artigo: 82,6 mm estimado vs 16,5 real)")
