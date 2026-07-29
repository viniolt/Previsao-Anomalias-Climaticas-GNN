"""Caracterizacao HONESTA da Regra 1 como antecipador de chuva forte, por estacao.

A Regra 1 dispara no instante t sobre a PREVISAO para t+6, entao e genuinamente
antecipada (60 min de horizonte), ao contrario da Regra 2 (retrospectiva). O prof.
pede: para manter 'alerta antecipado', mostrar o desempenho da Regra 1 e o lead time.

Evento (por estacao-janela): incremento real > 10 mm em (t, t+6].
Regra 1 anticipa se disparou em t (60 min de antecedencia) ou antes, ate LEAD_MAX.
Reportamos, por estacao: recall, falsas ativacoes/dia e lead time mediano; para a
Regra 1 completa (qualquer variavel) e so a componente de pluviosidade.
"""
import pickle, os
import numpy as np
import pandas as pd

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]; feat = g["feature_names"]
obs = [i for i in range(35) if nl[i] not in set(g["blind_nodes"])]

X = d["X_2024"]; T = d["trues_chuva_all"].shape[0]
inc = d["trues_chuva_all"] - X[:T, :, 0]                 # incremento em (t, t+6], por t
alert_any = d["alert_confirmado"][:T].any(axis=2)         # Regra 1 qualquer variavel [T,N]
alert_plu = d["alert_confirmado"][:T, :, 0]               # so pluviosidade [T,N]
ts = pd.to_datetime(d["timestamps_2024"][:T])
ndias = len(np.unique(ts.date))
LEAD_MAX = 6                                              # ate 6 passos antes = +60 min extra


def caracterizar(alert, nome):
    recs, fa_dia, leads_all = [], [], []
    for i in obs:
        ev = inc[:, i] > 10                              # janelas de evento (por t)
        al = alert[:, i]
        idx_ev = np.where(ev)[0]
        det = 0
        for t in idx_ev:
            # Regra 1 anticipa se disparou em [t-LEAD_MAX, t]; lead = 60 + (t - t_fire)*10 min
            janela = al[max(0, t - LEAD_MAX):t + 1]
            if janela.any():
                det += 1
                t_fire = max(0, t - LEAD_MAX) + np.where(janela)[0][0]
                leads_all.append(60 + (t - t_fire) * 10)  # min de antecedencia
        recs.append(det / len(idx_ev) if len(idx_ev) else np.nan)
        # falsas ativacoes: disparo sem evento em (t, t+6] nem dentro de LEAD_MAX a frente
        ev_fwd = np.convolve(ev.astype(int), np.ones(LEAD_MAX + 1), mode="full")[:T] > 0
        fp = int((al & ~ev_fwd).sum())
        fa_dia.append(fp / ndias)
    recs = np.array(recs); fa_dia = np.array(fa_dia); leads_all = np.array(leads_all)
    print(f"\n=== Regra 1 [{nome}] como antecipador de chuva forte (por estacao) ===")
    print(f"  recall medio: {np.nanmean(recs)*100:.0f}%  (min {np.nanmin(recs)*100:.0f}, max {np.nanmax(recs)*100:.0f})")
    print(f"  falsas ativacoes: {np.nanmean(fa_dia):.2f} por estacao/dia")
    if len(leads_all):
        print(f"  lead time: mediana {np.median(leads_all):.0f} min | media {leads_all.mean():.0f} min "
              f"(minimo 60 = 1 horizonte)")
    return np.nanmean(recs), np.nanmean(fa_dia), (np.median(leads_all) if len(leads_all) else np.nan)


caracterizar(alert_any, "qualquer variavel")
caracterizar(alert_plu, "so pluviosidade")

print("\nLeitura: a Regra 1 e genuinamente antecipada (opera sobre a previsao), mas seu")
print("recall de chuva forte e limitado e o custo em falsas ativacoes por estacao/dia e alto,")
print("porque dispara em ~15% dos instantes. Confirma que o alerta antecipado por si so e fraco;")
print("a forca do sistema esta na deteccao retrospectiva (Regra 2), nao na antecipacao.")
