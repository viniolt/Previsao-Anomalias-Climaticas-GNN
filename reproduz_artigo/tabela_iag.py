"""Tabela III do artigo: validacao do no cego 33 contra as medicoes reais do
IAG-USP (agregados diarios, 366 dias de 2024).

Fonte IAG: planilhas anuais da estacao (Prec: aba 'ano' com totais diarios;
T ar e UR: abas mensais hora x dia, media das horas disponiveis — a estacao
registra 7h-15h). Se as planilhas nao estiverem disponiveis, o script le o
CSV derivado `dados_iag_2024_diario.csv` (nesta pasta), que e gerado
automaticamente quando as planilhas sao encontradas.

Lado do modelo: serie do no 33 em X_2024 (Feature Propagation) e em
X_2024_baseline (media dos vizinhos, 1 iteracao). Precipitacao diaria =
soma dos incrementos positivos do acumulado; Temp/UR = media diaria.
"""
import os, unicodedata
import numpy as np
import pandas as pd
import pickle

_HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(_HERE), "artifacts")
CSV = os.path.join(_HERE, "dados_iag_2024_diario.csv")
MESES = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]


def achar_pasta_iag():
    """Procura a pasta com as planilhas do IAG subindo a partir do script."""
    base = _HERE
    for _ in range(4):
        base = os.path.dirname(base)
        if not os.path.isdir(base):
            continue
        for dnome in sorted(os.listdir(base)):
            if "IAG" in unicodedata.normalize("NFC", dnome).upper():
                cand = os.path.join(base, dnome, "iagUSP")
                if os.path.isdir(cand):
                    return cand
    return None


def parse_iag(pasta):
    """Devolve DataFrame diario (indice = data) com prec_mm, temp_c, ur_pct."""
    # precipitacao: aba 'ano' = matriz dia x mes de totais diarios
    xp = pd.ExcelFile(os.path.join(pasta, "Prec", "Pc2024b.xls"))
    an = xp.parse("ano", header=None)
    prec = {}
    for m in range(12):
        col = an.iloc[7:38, 2 + m].apply(pd.to_numeric, errors="coerce")
        for d, v in zip(range(1, 32), col):
            try:
                prec[pd.Timestamp(2024, m + 1, d)] = float(v)
            except ValueError:
                pass                                   # dia inexistente no mes
    # temp e UR: abas mensais, linhas 3..26 = horas 1..24, colunas 1.. = dias
    series = {"prec_mm": pd.Series(prec)}
    for chave, arq in [("temp_c", os.path.join("T ar", "2024.xls")),
                       ("ur_pct", os.path.join("UR", "2024.xls"))]:
        x = pd.ExcelFile(os.path.join(pasta, arq))
        vals = {}
        for m, aba in enumerate(MESES):
            df = x.parse(aba, header=None)
            bloco = df.iloc[3:27, 1:32].apply(pd.to_numeric, errors="coerce")
            dm = bloco.mean(axis=0, skipna=True)
            for j, v in enumerate(dm):
                if np.isfinite(v):
                    try:
                        vals[pd.Timestamp(2024, m + 1, j + 1)] = float(v)
                    except ValueError:
                        pass
        series[chave] = pd.Series(vals)
    return pd.DataFrame(series).sort_index()


if os.path.exists(CSV):
    iag = pd.read_csv(CSV, index_col=0, parse_dates=True)
else:
    pasta = achar_pasta_iag()
    if pasta is None:
        raise SystemExit("nem o CSV derivado nem as planilhas do IAG foram encontrados")
    iag = parse_iag(pasta)
    iag.to_csv(CSV)
    print(f"agregados diarios do IAG salvos em {os.path.basename(CSV)}")

d = pickle.load(open(os.path.join(ART, "X_test.pkl"), "rb"))
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
i33 = g["nodes_lista"].index("33")
ts = pd.to_datetime(d["timestamps_2024"])


def diarios(arr):
    plu = pd.Series(np.asarray(arr)[:, i33, 0], index=ts)
    prec = plu.diff().clip(lower=0).groupby(ts.date).sum()
    temp = pd.Series(np.asarray(arr)[:, i33, 1], index=ts).groupby(ts.date).mean()
    ur = pd.Series(np.asarray(arr)[:, i33, 2], index=ts).groupby(ts.date).mean()
    out = pd.DataFrame({"prec_mm": prec, "temp_c": temp, "ur_pct": ur})
    out.index = pd.to_datetime(out.index)
    return out


est = {"FP": diarios(d["X_2024"]), "MV": diarios(d["X_2024_baseline"])}

print("\n=== Tabela III: no cego 33 vs medicoes reais do IAG-USP (diario, 2024) ===")
print(f"{'Variavel':>14} {'Met.':>4} {'MAE':>8} {'RMSE':>8} {'r':>6}")
for var, nome in [("prec_mm", "Precipitacao"), ("temp_c", "Temperatura"), ("ur_pct", "Umid. rel.")]:
    for met in ["FP", "MV"]:
        j = pd.concat([iag[var], est[met][var]], axis=1, keys=["iag", "mod"]).dropna()
        e = j["mod"] - j["iag"]
        r = j["iag"].corr(j["mod"])
        print(f"{nome:>14} {met:>4} {e.abs().mean():8.2f} {np.sqrt((e**2).mean()):8.2f} {r:6.2f}")

print("\nMedias/totais anuais (contexto do texto):")
print(f"  temperatura: IAG {iag['temp_c'].mean():.2f} C | FP {est['FP']['temp_c'].mean():.2f} C")
print(f"  precipitacao total: IAG {iag['prec_mm'].sum():.0f} mm | FP {est['FP']['prec_mm'].sum():.0f} mm")
d13 = pd.Timestamp("2024-01-13")
print(f"  13/01/2024: IAG {iag.loc[d13, 'prec_mm']:.1f} mm | FP {est['FP'].loc[d13, 'prec_mm']:.1f} mm")
