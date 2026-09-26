"""Confere a leitura das datas dos CSVs do CGE antes de rodar o notebook.

Para cada arquivo de estacoes_meteorologicas/meteorologicas_<ano>/, converte a
coluna DATA como o notebook faz e exige que a data convertida seja igual ao texto
original (AAAA-MM-DD). Tambem conta quantas linhas a leitura antiga
(format="mixed", dayfirst=True) trocaria dia e mes nesta versao do pandas.

Uso (na raiz do repositorio):
    python reproduz_artigo/verifica_datas.py            # todos os anos presentes
    python reproduz_artigo/verifica_datas.py 2024       # so alguns anos

Sai com codigo 1 se alguma data for perdida ou trocada pela leitura atual.
"""
import glob
import os
import sys

import pandas as pd

from caminhos import dados_brutos


def coluna_data(arq):
    with open(arq, encoding="utf-8", errors="ignore") as f:
        cab = f.readline()
    sep = ";" if cab.count(";") > cab.count(",") else ","
    df = pd.read_csv(arq, sep=sep, usecols=[1], dtype=str, engine="python",
                     on_bad_lines="skip")
    return df.iloc[:, 0].dropna().str.strip()


def conferir(texto):
    esperado = texto.str[:10]
    atual = pd.to_datetime(texto, errors="coerce", format="ISO8601")
    antigo = pd.to_datetime(texto, errors="coerce", format="mixed", dayfirst=True)
    erros = int((atual.dt.strftime("%Y-%m-%d") != esperado).sum())
    trocas_antigo = int((antigo.dt.strftime("%Y-%m-%d") != esperado).sum())
    return erros, trocas_antigo


def main():
    base = dados_brutos()
    anos = sys.argv[1:] or sorted(d.split("_")[-1] for d in os.listdir(base)
                                  if d.startswith("meteorologicas_"))
    print(f"pandas {pd.__version__} | pasta: {base}\n")
    total_erros = 0
    for ano in anos:
        arquivos = sorted(glob.glob(os.path.join(base, f"meteorologicas_{ano}", "*.csv")))
        linhas = erros = trocas = 0
        for arq in arquivos:
            texto = coluna_data(arq)
            e, t = conferir(texto)
            linhas += len(texto); erros += e; trocas += t
            if e:
                print(f"  ERRO {os.path.basename(arq)}: {e} datas lidas errado")
        total_erros += erros
        print(f"{ano}: {len(arquivos)} arquivos, {linhas} linhas | leitura atual: "
              f"{erros} erros | leitura antiga trocaria: {trocas} ({trocas / max(linhas, 1):.0%})")
    if total_erros:
        print("\nFALHOU: ha datas lidas errado. Nao rode o notebook.")
        sys.exit(1)
    print("\nOK: todas as datas foram lidas corretamente.")


if __name__ == "__main__":
    main()
