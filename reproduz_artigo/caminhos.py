"""Caminhos canonicos do repositorio, resolvidos a partir da posicao do script.

Todos os scripts desta pasta ficam um nivel abaixo da raiz, entao a raiz e o
diretorio-pai. Centralizar os caminhos aqui evita que uma reorganizacao de
pastas quebre cada script separadamente.

Uso:
    from caminhos import ARTIFACTS, FIGS
"""
import os
import unicodedata

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)

ARTIFACTS = os.path.join(RAIZ, "artifacts")
RELATORIO = os.path.join(RAIZ, "academico", "IT-relatorio-final")
FIGS = os.path.join(RELATORIO, "figs")


def dados_brutos():
    """Pasta dos CSVs do CGE (estacoes_meteorologicas/), tolerante a acentos."""
    nome = next((d for d in os.listdir(RAIZ)
                 if "meteorol" in unicodedata.normalize("NFC", d)), None)
    if nome is None:
        raise FileNotFoundError("pasta de dados brutos nao encontrada na raiz")
    return os.path.join(RAIZ, nome)
