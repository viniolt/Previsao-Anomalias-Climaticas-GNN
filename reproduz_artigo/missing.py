import os, glob, io, re, unicodedata, sys, pickle
import numpy as np
import pandas as pd

# localiza a pasta de dados brutos (estações_meteorológicas) subindo a partir
# do script: no repositório ela fica na raiz, ao lado de reproduz_artigo/
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE)),
          os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "completo")]
BASE = None
for _root in _CANDS:
    if not os.path.isdir(_root):
        continue
    hit = next((d for d in os.listdir(_root) if "meteorol" in unicodedata.normalize("NFC", d)), None)
    if hit:
        BASE = os.path.join(_root, hit)
        break
if BASE is None:
    sys.exit("pasta estações_meteorológicas não encontrada acima do script")

_g = pickle.load(open(os.path.join(os.path.dirname(_HERE), "artifacts", "graph_data.pkl"), "rb"))
NODES = list(_g["nodes_lista"])
NOS_CEGOS = {"33", "34", "35"}


def normalizar_nome(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


MAPA_FIXO_CSV = {"p503": "19", "p510": "3", "p846": "14", "p1000857": "13", "p1000880": "0"}
_STOP = {"do","da","de","dos","das","e","o","a","os","as","em","no","na","sp","cge","dados",
         "pmsp","comdec","municipal","meteorologica","estacao","esta","paco","rinco",
         "fo","la","sa","cl","mo","mp","iq","vm","pr","pa","bt","pi","ip","sm","mb","af",
         "mg","ad","ja","cs","jt","pe","it","pj","vp"}


def _palavras(s):
    toks = [t for t in normalizar_nome(s).split("_") if not re.match(r"^p\d+$", t)]
    return [t for t in toks if len(t) >= 3 and t not in _STOP]


IDX = [{"node_id": n, "tokens": _palavras(_g["labels"].get(n, ""))} for n in NODES]


def achar_no(nome):
    stem = normalizar_nome(nome)
    for chave, nid in MAPA_FIXO_CSV.items():
        if chave in stem:
            return nid
    tc = _palavras(nome)
    best, bs = None, 0.0
    for it in IDX:
        tl = it["tokens"]
        if not tl:
            continue
        sc = sum(1 for pl in tl if any(pl in pc or pc in pl for pc in tc)) / len(tl)
        if sc > bs:
            bs, best = sc, it["node_id"]
    return best if bs >= 0.5 else None


def _sep(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        l = f.readline()
    return ";" if l.count(";") > l.count(",") else ","


def carregar(path):
    out = []
    for arq in glob.glob(os.path.join(path, "*.csv")):
        sid = os.path.basename(arq).replace(".csv", "")
        sep = _sep(arq)
        if sep == ",":
            with open(arq, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.readlines()
            if txt:
                txt[0] = txt[0].replace("DATA,,", "DATA,", 1).replace("data,,", "data,", 1)
            df = pd.read_csv(io.StringIO("".join(txt)), sep=sep, engine="python", on_bad_lines="skip")
        else:
            df = pd.read_csv(arq, sep=sep, engine="python", on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]
        if "DATA" not in df.columns:
            cand = [c for c in df.columns if c.lower() in ("data", "datetime", "timestamp")]
            if not cand:
                continue
            df = df.rename(columns={cand[0]: "DATA"})
        # datas ISO: format="mixed" + dayfirst=True troca dia/mes no pandas 3
        df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce", format="ISO8601")
        df = df.dropna(subset=["DATA"]).set_index("DATA").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        for col in df.columns:
            if df[col].dtype == "object":
                # equivalente ao errors="ignore" do notebook: normaliza o decimal e
                # converte quando a coluna inteira for numerica; caso contrario mantem.
                s = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
                conv = pd.to_numeric(s, errors="coerce")
                df[col] = conv if conv.notna().any() else s
        df["station_id"] = sid
        out.append(df)
    return pd.concat(out) if out else pd.DataFrame()


def primeira(df, cands):
    norm = {normalizar_nome(c): c for c in df.columns}
    for c in cands:
        if c in df.columns:
            return c
        if normalizar_nome(c) in norm:
            return norm[normalizar_nome(c)]
    return None


def analisar(ano):
    df = carregar(f"{BASE}/meteorologicas_{ano}")
    if df.empty:
        return None
    col_plu = primeira(df, ["PLU(mm)", "PLU (mm)", "PLU_mm", "pluviosidade(mm)", "Pluviosidade(mm)"])
    col_vel = primeira(df, ["Vel.VT(m/s)", "VEL(m/s)", "VEL", "Velocidade(m/s)"])
    col_temp = primeira(df, ["Temp(oC)", "Temp(°C)", "Temperatura(°C)", "Sens. Térmica(°C)"])
    col_umid = primeira(df, ["Umid.Rel(%)", "Umid Rel(%)", "Umidade(%)", "UMID(%)", "Umidade Relativa(%)"])

    df["PLU_real"] = pd.to_numeric(df[col_plu], errors="coerce") if col_plu else np.nan
    df["TEMP_real"] = pd.to_numeric(df[col_temp], errors="coerce") if col_temp else np.nan
    df["UMID_real"] = pd.to_numeric(df[col_umid], errors="coerce") if col_umid else np.nan
    df["VEL_real"] = pd.to_numeric(df[col_vel], errors="coerce") if col_vel else np.nan

    df["node_id"] = df["station_id"].astype(str).map({s: achar_no(s) for s in df["station_id"].astype(str).unique()})
    df = df.dropna(subset=["node_id"])
    df["node_id"] = df["node_id"].astype(str)

    feats = ["PLU_real", "TEMP_real", "UMID_real", "VEL_real"]
    piv = df.reset_index().pivot_table(index="DATA", columns="node_id", values=feats, aggfunc="mean").sort_index()
    piv = piv.resample("10min").mean()
    piv = piv.reindex(columns=pd.MultiIndex.from_product([feats, NODES]), fill_value=np.nan)

    obs = [n for n in NODES if n not in NOS_CEGOS]
    res = {"ano": ano, "instantes": len(piv)}
    for f in feats:
        bloco = piv[f][obs]
        res[f] = float(bloco.isna().to_numpy().mean() * 100)
    return res


anos = [int(a) for a in sys.argv[1:]] or [2024]
rows = [analisar(a) for a in anos]
print("\n=== % de células AUSENTES na grade de 10 min (apenas os 32 nós observados) ===")
print(f"{'ano':>6} {'instantes':>10} {'PLU':>8} {'TEMP':>8} {'UMID':>8} {'VEL':>8}")
for r in rows:
    if r:
        print(f"{r['ano']:>6} {r['instantes']:>10} {r['PLU_real']:>7.2f}% {r['TEMP_real']:>7.2f}% {r['UMID_real']:>7.2f}% {r['VEL_real']:>7.2f}%")
