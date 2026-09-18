"""Regenera `artifacts/X_test.pkl` (celula 44 do notebook) SEM retreinar.

O ano de teste (2024) e os pesos treinados (`model_plu.pt`, `model_temp.pt`),
o grafo (`graph_data.pkl`) e a climatologia (`tau.pkl`, `sigma_inv.pkl`) estao
versionados; so o tensor de teste e derivado deles falta no disco (>100 MB,
ignorado pelo .gitignore). Este script refaz exatamente as etapas do notebook
que dependem apenas de 2024:

  1. leitura dos CSVs de `estações_meteorológicas/meteorologicas_2024`;
  2. grade de 10 min + preenchimento em cascata (identico a celula 9);
  3. baseline (media simples dos vizinhos) e Feature Propagation R=40 (celula 15);
  4. inferencia dos dois GraphSAGE em resIduo -> previsao absoluta (celulas 19/27);
  5. Regra 1 (limiar mensal + janela anti-ruido) e Regra 2 (Mahalanobis) (celula 40);
  6. metricas por no e dos nos cegos (celulas 29/31/42) e gravacao do pickle.

A camada WeightedSAGEConv e reimplementada em PyTorch puro (index_add_), com a
mesma algebra da versao PyG do notebook, para nao exigir torch_geometric.

Uso: python reproduz_artigo/regen_x_test.py
"""
import glob
import io
import os
import pickle
import re
import unicodedata

import numpy as np
import pandas as pd
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(RAIZ, "artifacts")
DEVICE = torch.device("cpu")

HORIZON = 6
TARGET_IDX_PLU, TARGET_IDX_TEMP = 0, 1
FP_ITER = 40
W_ALERT = 3
NOS_SEM_DADOS = {"33", "34", "35"}
ANO_TESTE = 2024


# ---------------------------------------------------------------------------
# 1. Leitura bruta dos CSVs (celula 7 do notebook)
# ---------------------------------------------------------------------------
def _detectar_sep(caminho_csv):
    with open(caminho_csv, "r", encoding="utf-8", errors="ignore") as f:
        linha = f.readline()
    return ";" if linha.count(";") > linha.count(",") else ","


def _corrigir_cabecalho_csv_cge(caminho_csv):
    with open(caminho_csv, "r", encoding="utf-8", errors="ignore") as f:
        linhas = f.readlines()
    if not linhas:
        return ""
    linhas[0] = linhas[0].replace("DATA,,", "DATA,", 1).replace("data,,", "data,", 1)
    return "".join(linhas)


def carregar_estacoes_sem_agregar(path):
    arquivos = glob.glob(os.path.join(path, "*.csv"))
    estacoes = []
    print(f"Processando {len(arquivos)} arquivos em {path}")
    for arq in arquivos:
        id_estacao = os.path.basename(arq).replace(".csv", "")
        sep = _detectar_sep(arq)
        if sep == ",":
            df = pd.read_csv(io.StringIO(_corrigir_cabecalho_csv_cge(arq)), sep=sep,
                             engine="python", encoding="utf-8", on_bad_lines="skip")
        else:
            df = pd.read_csv(arq, sep=sep, engine="python", encoding="utf-8",
                             on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]
        if "DATA" not in df.columns:
            possiveis = [c for c in df.columns
                         if c.lower() in ("data", "datetime", "timestamp")]
            if not possiveis:
                raise ValueError(f"CSV {arq} sem coluna DATA: {df.columns.tolist()}")
            df = df.rename(columns={possiveis[0]: "DATA"})
        if "Unnamed: 2" in df.columns:
            raise ValueError(f"Cabecalho CGE quebrado em {os.path.basename(arq)}")
        df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce", format="mixed",
                                    dayfirst=True)
        df = df.dropna(subset=["DATA"]).set_index("DATA").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        for col in df.columns:
            if df[col].dtype == "object":
                s = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
                conv = pd.to_numeric(s, errors="coerce")
                df[col] = conv if conv.notna().any() else s
        df["station_id"] = id_estacao
        estacoes.append(df)
    if not estacoes:
        raise ValueError(f"Nenhum CSV em {path}")
    df_final = pd.concat(estacoes).sort_index()
    df_final.index.name = "DATA"
    return df_final


# ---------------------------------------------------------------------------
# 2. Montagem do tensor [T, N, F] (celula 9 do notebook)
# ---------------------------------------------------------------------------
def normalizar_nome(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


MAPA_FIXO_CSV = {"p503": "19", "p510": "3", "p846": "14",
                 "p1000857": "13", "p1000880": "0"}

_STOPWORDS_MATCH = {
    "do", "da", "de", "dos", "das", "e", "o", "a", "os", "as", "em", "no", "na",
    "sp", "cge", "dados", "pmsp", "comdec", "municipal", "meteorologica",
    "estacao", "esta", "paco", "rinco",
    "fo", "la", "sa", "cl", "mo", "mp", "iq", "vm", "pr", "pa", "bt", "pi",
    "ip", "sm", "mb", "af", "mg", "ad", "ja", "cs", "jt", "pe", "it", "pj", "vp",
}


def _palavras_relevantes(s):
    tokens = [t for t in normalizar_nome(s).split("_") if not re.match(r"^p\d+$", t)]
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS_MATCH]


def _score_tokens(tokens_csv, tokens_label):
    if not tokens_label:
        return 0.0
    return sum(1 for pl in tokens_label
               if any(pl in pc or pc in pl for pc in tokens_csv)) / len(tokens_label)


def encontrar_no_correspondente(nome_arquivo, indice_nos):
    stem_norm = normalizar_nome(nome_arquivo)
    for chave, node_id in MAPA_FIXO_CSV.items():
        if chave in stem_norm:
            return node_id
    tokens_csv = _palavras_relevantes(nome_arquivo)
    melhor_no, melhor_score = None, 0.0
    for item in indice_nos:
        score = _score_tokens(tokens_csv, item["tokens_label"])
        if score > melhor_score:
            melhor_score, melhor_no = score, item["node_id"]
    return melhor_no if melhor_score >= 0.5 else None


def _primeira_coluna_existente(df, candidatos):
    normalizadas = {normalizar_nome(c): c for c in df.columns}
    for c in candidatos:
        if c in df.columns:
            return c
    for c in candidatos:
        if normalizar_nome(c) in normalizadas:
            return normalizadas[normalizar_nome(c)]
    raise KeyError(f"Nenhuma das colunas {candidatos} existe: {df.columns.tolist()}")


def _selecionar_coluna_temperatura(df):
    candidatos = ["Temp(oC)", "Temp(°C)", "Temperatura(°C)",
                  "Sens. Térmica(°C)", "Sens. Termica(°C)"]
    normalizadas = {normalizar_nome(c): c for c in df.columns}
    disponiveis = []
    for cand in candidatos:
        col = cand if cand in df.columns else normalizadas.get(normalizar_nome(cand))
        if col is None:
            continue
        disponiveis.append((col, int(pd.to_numeric(df[col], errors="coerce").notna().sum())))
    if not disponiveis:
        raise KeyError("Nenhuma coluna de temperatura encontrada.")
    disponiveis.sort(key=lambda it: (-it[1], candidatos.index(it[0])
                                     if it[0] in candidatos else 999))
    return disponiveis[0][0]


def preparar_dataset_final_v6(df_input, nodes_lista, labels, nome_conjunto="2024"):
    df_fix = df_input.copy()
    col_plu = _primeira_coluna_existente(
        df_fix, ["PLU(mm)", "PLU (mm)", "PLU_mm", "pluviosidade(mm)", "Pluviosidade(mm)"])
    col_vel = _primeira_coluna_existente(
        df_fix, ["Vel.VT(m/s)", "VEL(m/s)", "VEL", "Velocidade(m/s)"])
    col_temp = _selecionar_coluna_temperatura(df_fix)
    col_umid = None
    for cand in ["Umid.Rel(%)", "Umid Rel(%)", "Umidade(%)", "UMID(%)",
                 "Umidade Relativa(%)"]:
        try:
            col_umid = _primeira_coluna_existente(df_fix, [cand])
            break
        except KeyError:
            continue
    print(f"[{nome_conjunto}] PLU={col_plu} TEMP={col_temp} UMID={col_umid} VEL={col_vel}")

    df_fix["PLU_real"] = pd.to_numeric(df_fix[col_plu], errors="coerce")
    df_fix["VEL_real"] = pd.to_numeric(df_fix[col_vel], errors="coerce")
    df_fix["TEMP_real"] = pd.to_numeric(df_fix[col_temp], errors="coerce")
    df_fix["UMID_real"] = (pd.to_numeric(df_fix[col_umid], errors="coerce")
                           if col_umid is not None else np.nan)

    indice_nos = [{"node_id": n, "tokens_label": _palavras_relevantes(labels[n])}
                  for n in nodes_lista]
    ids = sorted(set(df_fix["station_id"].astype(str).dropna().unique().tolist()))
    mapa = {sid: encontrar_no_correspondente(sid, indice_nos) for sid in ids}
    df_fix["node_id"] = df_fix["station_id"].astype(str).map(mapa)
    sem_no = sorted([s for s, n in mapa.items() if n is None])
    print(f"[{nome_conjunto}] estacoes mapeadas: {sum(v is not None for v in mapa.values())}"
          f"/{len(mapa)}" + (f" | sem no: {sem_no}" if sem_no else ""))

    df_fix = df_fix.dropna(subset=["node_id"]).copy()
    df_fix["node_id"] = df_fix["node_id"].astype(str)

    features = ["PLU_real", "TEMP_real", "UMID_real", "VEL_real"]
    df_pivot = (df_fix.reset_index()
                .pivot_table(index="DATA", columns="node_id", values=features,
                             aggfunc="mean")
                .sort_index())
    df_pivot = df_pivot.resample("10min").mean()
    df_pivot = df_pivot.reindex(
        columns=pd.MultiIndex.from_product([features, nodes_lista]), fill_value=np.nan)

    # nos cegos permanecem cegos, mesmo que alguma fonte apareca
    for feat in features:
        for no in NOS_SEM_DADOS:
            if no in df_pivot[feat].columns:
                df_pivot[(feat, no)] = np.nan

    df_pivot = df_pivot.interpolate(method="time", limit_direction="both")
    # nos cegos entram no FP inicializados em 0.0 (nao por media dos vizinhos)
    for feat in features:
        for no in NOS_SEM_DADOS:
            if no in df_pivot[feat].columns:
                df_pivot[(feat, no)] = 0.0
    df_pivot = df_pivot.ffill().bfill()
    for feat in features:
        if feat == "PLU_real":
            df_pivot[feat] = df_pivot[feat].fillna(0.0)
        else:
            media = np.nanmean(df_pivot[feat].to_numpy(dtype=float))
            df_pivot[feat] = df_pivot[feat].fillna(0.0 if np.isnan(media) else float(media))

    data_3d = (df_pivot.values
               .reshape(len(df_pivot), len(features), len(nodes_lista))
               .transpose(0, 2, 1))
    return torch.tensor(data_3d, dtype=torch.float32), df_pivot.index


# ---------------------------------------------------------------------------
# 3. Feature Propagation e baseline (celula 15)
# ---------------------------------------------------------------------------
def feature_propagation_batch(X, edge_index, edge_weight, known_mask, num_iter=FP_ITER):
    X = X.clone().float()
    X_known = X[:, known_mask, :].clone()
    src, dst = edge_index[0], edge_index[1]
    ew = edge_weight.float().view(1, -1, 1)
    T, N, F = X.shape
    for _ in range(num_iter):
        msgs = X[:, src, :] * ew
        X_new = torch.zeros(T, N, F, device=X.device)
        X_new.index_add_(1, dst, msgs)
        X_new[:, known_mask, :] = X_known
        X = X_new
    return X


def imputacao_media_vizinhos_batch(X, edge_index, blind_idx):
    X = X.clone().float()
    src, dst = edge_index[0], edge_index[1]
    T, N, F = X.shape
    cont = torch.zeros(N, device=X.device)
    cont.index_add_(0, dst, torch.ones(dst.size(0), device=X.device))
    cont = cont.clamp(min=1.0).view(1, N, 1)
    soma = torch.zeros(T, N, F, device=X.device)
    soma.index_add_(1, dst, X[:, src, :])
    media = soma / cont
    X[:, torch.tensor(blind_idx), :] = media[:, torch.tensor(blind_idx), :]
    return X


# ---------------------------------------------------------------------------
# 4. GraphSAGE ponderado em PyTorch puro (celula 19, sem PyG)
# ---------------------------------------------------------------------------
class WeightedSAGEConv(torch.nn.Module):
    """h'_i = W_self x_i + W_neigh * (sum_j w_ij x_j) / (sum_j w_ij)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
        self.lin_neigh = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, edge_weight)
        deg = deg.clamp(min=1e-8)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, edge_weight.view(-1, 1) * x[src])
        return self.lin_self(x) + self.lin_neigh(agg / deg.unsqueeze(-1))


class MeteorologiaGNN(torch.nn.Module):
    def __init__(self, in_feats=4, h_feats=64):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_feats, h_feats)
        self.conv2 = WeightedSAGEConv(h_feats, 1)

    def forward_batched(self, edge_index, edge_weight, x_bnf):
        B, N, Fd = x_bnf.shape
        E = edge_index.size(1)
        offs = (torch.arange(B, device=x_bnf.device) * N).repeat_interleave(E)
        ei_b = edge_index.repeat(1, B) + offs
        ew_b = edge_weight.repeat(B)
        h = self.conv1(x_bnf.reshape(B * N, Fd), ei_b, ew_b)
        h = torch.relu(h)
        h = self.conv2(h, ei_b, ew_b)
        return h.reshape(B, N)


def prever_absoluto(modelo, edge_index, edge_weight, X, target_idx, clip_min=None,
                    batch=1024):
    modelo.eval()
    outs = []
    with torch.no_grad():
        for a in range(0, len(X), batch):
            outs.append(modelo.forward_batched(edge_index, edge_weight, X[a:a + batch]))
    res = torch.cat(outs, dim=0)
    if clip_min is not None:
        res = res.clamp(min=clip_min)
    return X[:, :, target_idx] + res


# ---------------------------------------------------------------------------
# 5. Regras de deteccao (celula 40)
# ---------------------------------------------------------------------------
def aplicar_janela_antiruido(alert_series, w_alert=W_ALERT):
    alert_series = np.asarray(alert_series, dtype=bool)
    confirmado = np.zeros_like(alert_series)
    run = np.zeros(alert_series.shape[1:], dtype=int)
    for t in range(alert_series.shape[0]):
        run = np.where(alert_series[t], run + 1, 0)
        confirmado[t] = run >= w_alert
    return confirmado


def main():
    print("=" * 70)
    g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
    nodes_lista = list(g["nodes_lista"])
    labels = g["labels"]
    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
    edge_weight = torch.tensor(g["edge_weight"], dtype=torch.float32)
    feature_names = list(g["feature_names"])
    blind_idx = [i for i, n in enumerate(nodes_lista) if str(n) in NOS_SEM_DADOS]
    known_mask = torch.tensor([str(n) not in NOS_SEM_DADOS for n in nodes_lista],
                              dtype=torch.bool)
    N_NODES = len(nodes_lista)
    print(f"grafo: {N_NODES} nos, {edge_index.size(1)} arestas direcionadas, "
          f"cegos={sorted(NOS_SEM_DADOS)}")

    base = next(os.path.join(RAIZ, d) for d in os.listdir(RAIZ)
                if "meteorol" in unicodedata.normalize("NFC", d))
    df_2024 = carregar_estacoes_sem_agregar(os.path.join(base, f"meteorologicas_{ANO_TESTE}"))
    X_2024, timestamps_2024 = preparar_dataset_final_v6(df_2024, nodes_lista, labels)
    print("X_2024 (pre-FP):", tuple(X_2024.shape))

    X_2024_baseline = imputacao_media_vizinhos_batch(X_2024, edge_index, blind_idx)
    X_2024 = feature_propagation_batch(X_2024, edge_index, edge_weight, known_mask)
    print(f"Feature Propagation ({FP_ITER} it.) aplicado.")
    for j in blind_idx:
        print(f"  no cego {nodes_lista[j]}: PLU medio={X_2024[:, j, 0].mean():.4f} | "
              f"Temp media={X_2024[:, j, 1].mean():.4f}")

    X_test = X_2024[:-HORIZON, :, :]
    Y_test = X_2024[HORIZON:, :, TARGET_IDX_PLU]
    Y_test_temp = X_2024[HORIZON:, :, TARGET_IDX_TEMP]

    modelo = MeteorologiaGNN(4, g.get("model_h_feats", 64))
    modelo.load_state_dict(torch.load(os.path.join(ART, "model_plu.pt"),
                                      map_location=DEVICE))
    modelo_temp = MeteorologiaGNN(4, g.get("model_h_feats", 64))
    modelo_temp.load_state_dict(torch.load(os.path.join(ART, "model_temp.pt"),
                                           map_location=DEVICE))
    n_par = sum(p.numel() for p in modelo.parameters())
    print(f"modelos carregados | parametros por modelo: {n_par}")

    preds_chuva_all = prever_absoluto(modelo, edge_index, edge_weight, X_test,
                                      TARGET_IDX_PLU, clip_min=0.0).numpy()
    preds_temp_all = prever_absoluto(modelo_temp, edge_index, edge_weight, X_test,
                                     TARGET_IDX_TEMP).numpy()
    trues_chuva_all = Y_test.numpy()
    trues_temp_all = Y_test_temp.numpy()

    def _df_resultados(preds, trues, nome, criterio, window=120):
        linhas = []
        for j in range(preds.shape[1]):
            p, tr = preds[:, j], trues[:, j]
            idx_central = (int(np.argmax(tr)) if criterio == "pico_real"
                           else int(np.argmax(np.abs(tr - np.mean(tr)))))
            linhas.append({
                "no": j, "variavel": nome,
                "mae": float(np.mean(np.abs(p - tr))),
                "rmse": float(np.sqrt(np.mean((p - tr) ** 2))),
                "idx_central": idx_central,
                "start": max(0, idx_central - window // 2),
                "end": min(len(tr), idx_central + window // 2),
            })
        return pd.DataFrame(linhas)

    df_resultados_chuva = _df_resultados(preds_chuva_all, trues_chuva_all,
                                         "Pluviosidade", "pico_real")
    df_resultados_temp = _df_resultados(preds_temp_all, trues_temp_all,
                                        "Temperatura", "maior_desvio_media")

    tau = pickle.load(open(os.path.join(ART, "tau.pkl"), "rb"))
    sigma_inv = pickle.load(open(os.path.join(ART, "sigma_inv.pkl"), "rb"))

    X_hat_mv = X_test.clone()
    X_hat_mv[:, :, TARGET_IDX_PLU] = torch.tensor(preds_chuva_all)
    X_hat_mv[:, :, TARGET_IDX_TEMP] = torch.tensor(preds_temp_all)
    X_real_next = X_2024[HORIZON:]

    tempo_test = pd.to_datetime(list(timestamps_2024))[HORIZON:]
    meses_test = np.array([t.month for t in tempo_test])
    xhat_np = X_hat_mv.numpy()
    alert_raw = np.zeros((len(X_hat_mv), N_NODES, len(feature_names)), dtype=bool)
    for t in range(len(X_hat_mv)):
        alert_raw[t] = xhat_np[t] > tau[meses_test[t] - 1]
    alert_confirmado = aplicar_janela_antiruido(alert_raw)
    print(f"Regra 1: {alert_raw.sum()} ativacoes brutas -> "
          f"{alert_confirmado.sum()} confirmadas (W={W_ALERT})")

    e = X_real_next.numpy() - xhat_np                       # [T, N, F]
    anomaly_scores = np.sqrt(np.einsum("tnf,nfg,tng->tn", e, sigma_inv, e).clip(min=0))
    print(f"anomaly_scores: {anomaly_scores.shape} | AS medio: "
          f"{float(np.nanmean(anomaly_scores)):.4f}")

    ts_idx, no_idx, ft_idx = np.where(alert_confirmado)
    df_alertas = pd.DataFrame([{
        "Estacao": str(nodes_lista[n]),
        "Variavel": feature_names[fd],
        "Tipo": "Regra 1 (limiar previsto)",
        "AS_Score": float(anomaly_scores[t, n]),
        "Timestamp": tempo_test[t],
    } for t, n, fd in zip(ts_idx, no_idx, ft_idx)])

    X2024_np = X_2024.numpy()
    X2024_base_np = X_2024_baseline.numpy()
    df_metricas_cegos = pd.DataFrame([{
        "No": str(nodes_lista[j]), "Variavel": fname,
        "MAE_baseline_vs_FP": float(np.mean(np.abs(
            X2024_base_np[:, j, fd] - X2024_np[:, j, fd]))),
        "RMSE_baseline_vs_FP": float(np.sqrt(np.mean(
            (X2024_base_np[:, j, fd] - X2024_np[:, j, fd]) ** 2))),
    } for j in blind_idx for fd, fname in enumerate(feature_names)])

    resumo_pred = pd.DataFrame([
        {"Variavel": "Pluviosidade",
         "MAE": float(df_resultados_chuva["mae"].mean()),
         "RMSE": float(df_resultados_chuva["rmse"].mean())},
        {"Variavel": "Temperatura",
         "MAE": float(df_resultados_temp["mae"].mean()),
         "RMSE": float(df_resultados_temp["rmse"].mean())},
    ])
    print("\nMetricas medias de previsao (todos os 35 nos):")
    print(resumo_pred.to_string(index=False))

    obs = [i for i, n in enumerate(nodes_lista) if str(n) not in NOS_SEM_DADOS]
    print("\nAgregado nos 32 nos observados (convencao do artigo):")
    for nome, preds, trues in [("PLU", preds_chuva_all, trues_chuva_all),
                               ("Temp", preds_temp_all, trues_temp_all)]:
        d = preds[:, obs] - trues[:, obs]
        print(f"  {nome}: MAE={np.mean(np.abs(d)):.4f} "
              f"RMSE={np.sqrt(np.mean(d ** 2)):.4f}")

    export = {
        "X_2024": X_2024.numpy(),
        "X_2024_baseline": X_2024_baseline.numpy(),
        "timestamps_2024": [pd.Timestamp(t).isoformat() for t in timestamps_2024],
        "X_hat_mv": xhat_np,
        "X_real_next": X_real_next.numpy(),
        "alert_confirmado": alert_confirmado,
        "anomaly_scores": anomaly_scores,
        "preds_chuva_all": preds_chuva_all,
        "trues_chuva_all": trues_chuva_all,
        "preds_temp_all": preds_temp_all,
        "trues_temp_all": trues_temp_all,
        "df_alertas": df_alertas,
        "df_resultados_chuva": df_resultados_chuva,
        "df_resultados_temp": df_resultados_temp,
        "df_metricas_cegos": df_metricas_cegos,
        "resumo_pred": resumo_pred,
    }
    destino = os.path.join(ART, "X_test.pkl")
    with open(destino, "wb") as f:
        pickle.dump(export, f)
    print(f"\nX_test.pkl gravado em {destino} "
          f"({os.path.getsize(destino) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
