"""Reforco da avaliacao de deteccao (pedidos do prof. Ivan), tudo a partir dos
artefatos ja salvos, sem retreinar.

Produz:
  1. Curva precisao-revocacao sobre varios percentis + F1/especificidade.
  2. Detector CLIMATOLOGICO sem GraphSAGE: Mahalanobis sobre o erro de PERSISTENCIA
     (e = x_{t+h} - x_t) em todas as variaveis. Se detectar tao bem quanto o do
     modelo, o GraphSAGE nao agrega a deteccao.
  3. Bootstrap da revocacao/precisao no p99 (IC 95%).
  4. Antecedencia da Regra 1 (alert_confirmado) em relacao ao inicio do evento.
"""
import pickle, os
import numpy as np
import pandas as pd

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
sigma_inv = pickle.load(open(os.path.join(ART, "sigma_inv.pkl"), "rb"))  # [N,F,F]
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]; blind = set(g["blind_nodes"])
obs = [i for i in range(35) if nl[i] not in blind]

X = d["X_2024"]; T = d["trues_chuva_all"].shape[0]
ts = pd.to_datetime(d["timestamps_2024"])
t_alvo = pd.DatetimeIndex(ts[6:T + 6])
dias = pd.Series(t_alvo.date)

# ---- eventos: dia com incremento real > 10mm em algum posto obs (por t+6) ----
inc = d["trues_chuva_all"] - X[:T, :, 0]
ev = pd.DataFrame({"d": dias.values, "m": inc[:, obs].max(axis=1)}).groupby("d")["m"].max()
evento = (ev > 10).values
n_ev = int(evento.sum())
print(f"eventos: {n_ev} de {len(evento)} dias\n")

AS_model = d["anomaly_scores"]                      # [T,N] com GraphSAGE p/ PLU,Temp

# ---- detector climatologico sem grafo: Mahalanobis do erro de persistencia ----
e_persist = d["X_real_next"] - X[:T]                # x_{t+h} - x_t, [T,N,4]
AS_persist = np.zeros((T, 35))
for i in range(35):
    ei = e_persist[:, i, :]                         # [T,4]
    AS_persist[:, i] = np.sqrt(np.einsum("tf,fg,tg->t", ei, sigma_inv[i], ei).clip(min=0))


def metricas(AS, p):
    S = AS[:, obs]
    thr = np.nanpercentile(S, p)
    smax = pd.DataFrame({"d": dias.values, "m": S.max(axis=1)}).groupby("d")["m"].max().values
    flag = smax > thr
    tp = int((flag & evento).sum()); fp = int((flag & ~evento).sum())
    fn = int((~flag & evento).sum()); tn = int((~flag & ~evento).sum())
    rec = tp / (tp + fn) if tp + fn else 0
    pre = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0
    esp = tn / (tn + fp) if tn + fp else 0
    return rec, pre, f1, esp, int(flag.sum())


print("=== 1. Curva precisao-revocacao + F1 + especificidade (modelo com GraphSAGE) ===")
print(f"{'percentil':>10} {'recall':>8} {'precisao':>9} {'F1':>7} {'espec.':>8} {'dias sinal.':>12}")
for p in [90, 95, 97.5, 99, 99.5, 99.9]:
    rec, pre, f1, esp, ns = metricas(AS_model, p)
    print(f"{p:>10} {rec*100:7.1f}% {pre*100:8.1f}% {f1:7.3f} {esp*100:7.1f}% {ns:>12}")

print("\n=== 2. Detector climatologico SEM GraphSAGE (persistencia + Mahalanobis) ===")
print(f"{'percentil':>10} {'recall':>8} {'precisao':>9} {'F1':>7} {'espec.':>8}")
for p in [99, 99.9]:
    rec, pre, f1, esp, ns = metricas(AS_persist, p)
    print(f"{p:>10} {rec*100:7.1f}% {pre*100:8.1f}% {f1:7.3f} {esp*100:7.1f}%")
print("  (comparar com o modelo acima: se igual, o GraphSAGE nao agrega a deteccao)")

print("\n=== 3. Bootstrap IC95% da revocacao/precisao no p99 (modelo) ===")
S = AS_model[:, obs]
thr = np.nanpercentile(S, 99)
smax = pd.DataFrame({"d": dias.values, "m": S.max(axis=1)}).groupby("d")["m"].max().values
flag = smax > thr
rng = np.random.default_rng(0)
recs, pres = [], []
idx = np.arange(len(evento))
for _ in range(2000):
    b = rng.choice(idx, len(idx), replace=True)
    fl, ev_b = flag[b], evento[b]
    tp = (fl & ev_b).sum(); fp = (fl & ~ev_b).sum(); fn = (~fl & ev_b).sum()
    recs.append(tp / (tp + fn) if tp + fn else 0)
    pres.append(tp / (tp + fp) if tp + fp else 0)
print(f"  recall  p99: {np.mean(recs)*100:.1f}%  IC95 [{np.percentile(recs,2.5)*100:.1f}, {np.percentile(recs,97.5)*100:.1f}]")
print(f"  precis. p99: {np.mean(pres)*100:.1f}%  IC95 [{np.percentile(pres,2.5)*100:.1f}, {np.percentile(pres,97.5)*100:.1f}]")

print("\n=== 4. Antecedencia da Regra 1 (alert_confirmado) vs inicio do evento ===")
# Regra 1 confirmada em qualquer variavel/estacao observada, indexada por t (emissao).
alert = d["alert_confirmado"][:, obs, :].any(axis=(1, 2))     # [T] bool por instante de emissao
# inicio do evento no dia = primeiro instante (por t+6) com incremento>10 em algum obs
inc_obs_max = inc[:, obs].max(axis=1)                          # [T]
leads = []
dia_arr = np.array(dias.values)
for dd in ev.index[evento]:
    mask_dia = dia_arr == dd
    idxs = np.where(mask_dia & (inc_obs_max > 10))[0]
    if len(idxs) == 0:
        continue
    t_onset = idxs[0]                     # posicao no vetor de alvos (t+6)
    # alertas da Regra 1 emitidos ANTES do alvo t_onset (emissao <= t_onset, alvo = emissao+6)
    fired = np.where(alert[:t_onset + 1])[0]
    fired_dia = fired[dia_arr[np.clip(fired + 6, 0, T - 1)] == dd] if len(fired) else fired
    if len(fired_dia):
        lead_min = (t_onset - fired_dia[-1] + 6) * 10   # min de antecedencia do ultimo alerta antes do onset
        leads.append(max(lead_min, 0))
if leads:
    leads = np.array(leads)
    print(f"  eventos com alerta da Regra 1 antes do inicio: {len(leads)}/{n_ev}")
    print(f"  antecedencia mediana: {np.median(leads):.0f} min | media: {leads.mean():.0f} min | max: {leads.max():.0f} min")
else:
    print("  nenhum alerta da Regra 1 precedeu o inicio (medicao a revisar)")
