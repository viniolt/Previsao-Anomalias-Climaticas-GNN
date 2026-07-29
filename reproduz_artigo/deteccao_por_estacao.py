"""Item 1 do prof.: deteccao POR ESTACAO e POR JANELA de 60 min (nao por dia/rede),
com tolerancia temporal +-30 min, e metricas por estacao (recall, precisao, F1,
especificidade, falsas ativacoes por dia). Tudo dos artefatos, sem retreinar.

Evento por estacao-janela: a estacao i tem, no instante-alvo t+6, incremento real
> 10 mm em 60 min. Deteccao: o anomaly score da estacao i cruza o limiar p em
alguma janela dentro de +-TOL do instante do evento (tolerancia temporal).
"""
import pickle, os
import numpy as np
import pandas as pd

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nl = g["nodes_lista"]; lab = g["labels"]; blind = set(g["blind_nodes"])
obs = [i for i in range(35) if nl[i] not in blind]

X = d["X_2024"]; T = d["trues_chuva_all"].shape[0]
AS = d["anomaly_scores"]                       # [T, N]
inc = d["trues_chuva_all"] - X[:T, :, 0]       # incremento 60 min por estacao
TOL = 3                                          # +-3 passos de 10 min = +-30 min


def metricas_por_estacao(p):
    thr = np.nanpercentile(AS[:, obs], p)       # limiar global sobre estacoes obs
    rec, pre, f1, esp, fa_dia = [], [], [], [], []
    ts = pd.to_datetime(d["timestamps_2024"][6:T + 6])
    ndias = len(np.unique(ts.date))
    for i in obs:
        ev = inc[:, i] > 10                      # janelas de evento (por t+6)
        al = AS[:, i] > thr                      # ativacoes do score
        # deteccao com tolerancia: evento coberto se ha ativacao em +-TOL
        al_tol = np.convolve(al, np.ones(2 * TOL + 1), mode="same") > 0
        tp = int((ev & al_tol).sum()); fn = int((ev & ~al_tol).sum())
        # ativacoes que nao caem perto de nenhum evento = falsos positivos
        ev_tol = np.convolve(ev, np.ones(2 * TOL + 1), mode="same") > 0
        fp = int((al & ~ev_tol).sum()); tn = int((~al & ~ev_tol).sum())
        rec.append(tp / (tp + fn) if tp + fn else np.nan)
        pre.append(tp / (tp + fp) if tp + fp else np.nan)
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
        esp.append(tn / (tn + fp) if tn + fp else np.nan)
        fa_dia.append(fp / ndias)
    return (np.array(rec), np.array(pre), np.array(f1), np.array(esp), np.array(fa_dia))


print("=== Deteccao POR ESTACAO e POR JANELA de 60 min (tolerancia +-30 min) ===")
print("Evento = estacao com incremento real > 10 mm/60 min na janela.\n")
print(f"{'percentil':>10} {'recall':>16} {'precisao':>16} {'F1':>16} {'falsas/dia':>12}")
for p in [99, 99.9, 99.99]:
    rec, pre, f1, esp, fa = metricas_por_estacao(p)
    def ms(a): return f"{np.nanmean(a)*100:5.1f}% ({np.nanmin(a)*100:.0f}-{np.nanmax(a)*100:.0f})"
    print(f"{p:>10} {ms(rec):>16} {ms(pre):>16} {ms(f1):>16} {np.nanmean(fa):>11.2f}")

print("\nInterpretacao: comparar com a avaliacao por dia/rede (recall 100%/prec 27% no p99).")
print("A avaliacao por estacao-janela e muito mais dura: a precisao despenca porque")
print("cada ativacao fora de +-30 min de um evento conta como falso positivo.")

# numeros para o texto
rec99, pre99, f199, esp99, fa99 = metricas_por_estacao(99)
rec999, pre999, f1999, esp999, fa999 = metricas_por_estacao(99.9)
print("\n--- para o artigo (media sobre as 32 estacoes) ---")
print(f"p99   : recall {np.nanmean(rec99)*100:.0f}% | precisao {np.nanmean(pre99)*100:.1f}% | "
      f"F1 {np.nanmean(f199):.3f} | falsas ativacoes/dia {np.nanmean(fa99):.1f}")
print(f"p99.9 : recall {np.nanmean(rec999)*100:.0f}% | precisao {np.nanmean(pre999)*100:.1f}% | "
      f"F1 {np.nanmean(f1999):.3f} | falsas ativacoes/dia {np.nanmean(fa999):.1f}")
