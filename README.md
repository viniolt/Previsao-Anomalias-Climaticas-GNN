# Previsão de Anomalias Climáticas com GNN

> Monitoramento inteligente de estações meteorológicas em um Smart Campus com Graph Neural Networks.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.7-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)
![CUDA](https://img.shields.io/badge/GPU-CUDA_13-76b900?logo=nvidia)

---

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Quickstart](#quickstart)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Notebook — grafo-0.8.0-pyg.ipynb](#notebook--grafo-080-pygipynb)
- [Dashboard — dashboard.py](#dashboard--dashboardpy)
- [Datasets](#datasets)
- [Plano de Melhorias](#plano-de-melhorias)
- [Referências](#referências)

---

## Sobre o Projeto

Este projeto aplica **Graph Neural Networks (GraphSAGE)** para prever e detectar anomalias climáticas na rede de estações meteorológicas de São Paulo, integrando dados do **CGE** (Centro de Gerenciamento de Emergências), **INMET** e **IAG-USP**.

O grafo modela as estações como nós e as relações de proximidade geográfica como arestas ponderadas. A detecção de anomalias usa dois mecanismos complementares: limiar de percentil histórico (Regra 1) e distância de Mahalanobis multivariada (Regra 2).

**Ambiente de desenvolvimento principal:** Ubuntu 24.04 + RTX 3070 (CUDA). Suporte secundário a macOS com Apple M2 (Metal/MPS).

> **Atualização (julho/2026):** protocolo com validação retida (treino 2016–2022,
> validação 2023, teste 2024), grade de 10 min e horizonte t+6 (60 min); grafo
> `003.graphml` com as estações do INMET corrigidas; ablação com 3 sementes
> (MLP / GraphSAGE sem pesos / ponderado) e um **STGCN** com janela de 2 h,
> único modelo a superar a persistência no RMSE de pluviosidade; dashboard em
> duas versões (simples e técnica); artigo completo em `artigo/` e scripts de
> reprodução em `reproduz_artigo/`.

---

## Arquitetura do Sistema

```
estações_meteorológicas/
  meteorologicas_2016..2022/  ──► treino
  meteorologicas_2023/        ──► validação retida
  meteorologicas_2024/        ──► teste
003.graphml (grafo, 35 nós / 85 arestas)
        │
        ▼
grafo-0.8.0-pyg.ipynb
  ├─ Carregamento e alinhamento na grade de 10 min
  ├─ Feature Propagation nos nós cegos (Rossi et al., ICLR 2022)
  ├─ Treino do GraphSAGE (PLU + Temp, resíduo t+6 = 60 min)
  ├─ Inferência e avaliação (2024)
  └─ Detecção de anomalias (Regra 1 + Regra 2)
        │
        ▼
    artifacts/
  (modelos, tensores, limiares, métricas)
        │
        ├──► dashboard.py  (Streamlit)
        └──► reproduz_artigo/  (tabelas, figuras, ablação e STGCN do artigo)
```

**Nós cegos** (sem dados próprios): `33` IAG-USP · `34` INMET automático · `35` INMET convencional.
Estimados via Feature Propagation iterativa (interpolação harmônica, 40 iterações).

---

## Quickstart

### Pré-requisitos

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

| Dependência | Versão testada |
|---|---|
| Python | 3.12 |
| PyTorch | 2.11.0+cu130 |
| PyTorch Geometric | 2.7.0 |
| Streamlit | qualquer recente |
| NetworkX / pandas / numpy / matplotlib | qualquer |

### 1 — Rodar o notebook (treino + geração dos artefatos)

```bash
jupyter notebook grafo-0.8.0-pyg.ipynb
```

Execute todas as células em ordem (`Run All`). O treino leva ~10–15 min com GPU.
Ao final, a pasta `artifacts/` é criada com os 6 arquivos necessários para o dashboard.

### 2 — Iniciar o dashboard

```bash
streamlit run dashboard.py
```

Acesse em: **http://localhost:8501**

> O dashboard não precisa re-treinar o modelo — lê diretamente os artefatos gerados.

---

## Estrutura do Repositório

```
.
├── grafo-0.8.0-pyg.ipynb        # Notebook principal (versão atual: val-split, t+6)
├── grafo-0.8.0-pyg_backup_*.ipynb  # Estados anteriores da 0.8.0 (histórico)
├── grafo-0.7.x-pyg.ipynb        # Versões anteriores (histórico experimental)
├── dashboard.py                 # Painel Streamlit — versão simples (usuário final)
├── dashboard_tecnico_backup.py  # Painel técnico completo (5 abas, métricas)
├── 003.graphml                  # Grafo atual (35 nós, 85 arestas; INMET corrigido)
├── 001.graphml / 002.graphml    # Versões anteriores do grafo
├── analise_exploratoria_0.1.ipynb
├── artigo/                      # Artigo Latinoware 2026 (LaTeX + PDF + figuras)
├── reproduz_artigo/             # Scripts que reproduzem tabelas, figuras,
│                                #   ablação, validação retida e STGCN (ver LEIA-ME)
├── diagnostico_chuva/           # Demonstração acumulado vs. incremento
├── estações_meteorológicas/     # CSVs das estações CGE (2023 e 2024; 2016–2022 no Release)
├── graficos_2024/               # Gráficos de pluviosidade e temperatura por nó
│   ├── pluviosidade/
│   └── temperatura/
├── NÓS CEGOS/                   # Análises dos nós sem dados próprios
├── slidesGNN/                   # Slides e documentação técnica do projeto
├── docs/
├── exemplos/
└── artifacts/                   # Gerado pelo notebook; versionados apenas os
                                 #   pequenos (modelos, grafo, limiares, JSONs)
```

---

## Notebook — grafo-0.8.0-pyg.ipynb

### O que há de novo na versão 0.8.0

A versão 0.8.0 introduz três melhorias em relação à 0.7.5, onde pela primeira vez obtivemos resultados realistas para os nós cegos (estações 33, 34 e 35):

| # | Melhoria | Detalhe |
|---|---|---|
| 1 | Normalização das arestas por nó de destino (D⁻¹A) | Evita que arestas com peso dominante colapsen os demais |
| 2 | Feature Propagation iterativa nos nós cegos | Interpolação harmônica ótima (Rossi et al., ICLR 2022) |
| 3 | Detecção de anomalias dupla | Regra 1 (percentil) + Regra 2 (Mahalanobis) |

---

### Estrutura do notebook (célula a célula)

**Células 1–2 — Importações e seleção de dispositivo**
Define `selecionar_dispositivo()`, que escolhe automaticamente entre CUDA, MPS e CPU.

**Células 3–5 — Carregamento do grafo**
Lê `003.graphml` via NetworkX (coordenadas reais em `_COORDS_DEC` no próprio notebook). Funções auxiliares: `dms_to_decimal()` e `haversine()`.

**Células 6–9 — Leitura dos CSVs**
Detecta separador (`,`/`;`) automaticamente, corrige cabeçalhos malformados e mapeia cada CSV ao nó do grafo.

**Células 10–11 — Tensor `[T, N, F]`**
Monta o tensor principal com forma `[timesteps, nós, atributos]`:
- `T` = timesteps a cada 10 min (2016 a 2024)
- `N` = nós do grafo
- `F` = 4 atributos: `PLU`, `Temp`, `Umid.Rel.`, `Vel.VT`

---

### Mudança 1 — Normalização das arestas (célula 12)

**Problema anterior:** normalização min-max global fazia uma única aresta de peso alto dominar toda a agregação.

**Solução:** normalização por nó de destino (D⁻¹A):

$$\tilde{w}_{ij} = \frac{w_{ij}}{\sum_{k \in \mathcal{N}(j)} w_{kj}}$$

Versão simétrica opcional (D⁻¹/²AD⁻¹/²):

$$\tilde{w}_{ij}^{\text{sym}} = \frac{w_{ij}}{\sqrt{d_i \cdot d_j}}$$

---

### Mudança 2 — Feature Propagation nos nós cegos (célula 15)

Baseado em **Rossi et al., ICLR 2022**. O algoritmo minimiza a energia de Dirichlet do sinal no grafo:

$$\ell(x, G) = \frac{1}{2} x^\top \Delta x = \frac{1}{2} \sum_{ij} \tilde{a}_{ij} (x_i - x_j)^2$$

A cada iteração: (1) difusão $X \leftarrow \tilde{A}X$ e (2) restauração dos nós com dados reais. Após 40 iterações, os nós cegos convergem para a interpolação harmônica ótima.

Implementação vetorizada (todos os T timesteps em paralelo):

```python
def feature_propagation_batch(X, edge_index, edge_weight, known_mask, num_iter=40):
    X = X.clone().float()
    X_known = X[:, known_mask, :].clone()
    src, dst = edge_index[0], edge_index[1]
    ew = edge_weight.float().view(1, -1, 1)  # [1, E, 1]
    T, N, F = X.shape
    for _ in range(num_iter):
        msgs = X[:, src, :] * ew            # [T, E, F]
        X_new = torch.zeros(T, N, F)
        X_new.index_add_(1, dst, msgs)
        X_new[:, known_mask, :] = X_known   # restaura âncoras
        X = X_new
    return X
```

---

### Modelo GraphSAGE (células 18–25)

Arquitetura `MeteorologiaGNN`:

```
Entrada: x ∈ ℝ^{N×4}
  └─ WeightedSAGEConv(4 → 64)  + ReLU
  └─ WeightedSAGEConv(64 → 1)
Saída: ŷ ∈ ℝ^N
```

Cada camada implementa:

$$h'_i = W_{\text{self}} \cdot x_i + W_{\text{neigh}} \cdot \frac{\sum_{j \in \mathcal{N}(i)} w_{ij} \cdot x_j}{\sum_j w_{ij}}$$

- Dois modelos separados: **Pluviosidade** e **Temperatura**
- Treino: 2016–2022 · Validação retida: 2023 · Teste: 2024
- Loss: MSE apenas sobre nós com dados reais (nós cegos excluídos via máscara)
- Otimizador: Adam, lr=0.001, até 300 épocas, early stopping (paciência=25) e ReduceLROnPlateau, ambos na perda de validação

---

### Mudança 3 — Detecção de anomalias (célula 41)

**Regra 1 — Limiar meteorológico previsto**

Percentil 95 histórico por estação, mês e variável:

$$\tau_{i,m,k} = \text{P}_{95}\bigl(x_{i,k} \mid \text{mês} = m\bigr)$$

Alerta confirmado quando $\hat{x}_{i,k,t} > \tau_{i,m,k}$ por $W_{\text{alert}} = 3$ timesteps consecutivos (janela anti-ruído).

**Regra 2 — Anomaly Score multivariado (Mahalanobis)**

$$AS_{i,t} = \sqrt{e_{i,t}^\top \, \Sigma_i^{-1} \, e_{i,t}}, \quad e_{i,t} = x_{i,t+1} - \hat{x}_{i,t+1}$$

Captura padrões multivariados atípicos mesmo quando nenhuma variável individualmente ultrapassa seu limiar.

---

### Artefatos exportados (célula 45)

| Arquivo | Conteúdo |
|---|---|
| `model_plu.pt` | Pesos do modelo de pluviosidade |
| `model_temp.pt` | Pesos do modelo de temperatura |
| `tau.pkl` | Limiares τ `[12, N, F]` — percentil 95 por mês |
| `sigma_inv.pkl` | Inversas de covariância `[N, F, F]` para Mahalanobis |
| `graph_data.pkl` | Nós, arestas, coordenadas e nomes das estações |
| `X_test.pkl` | Tensores de teste, previsões, alertas e métricas |

> A pasta `artifacts/` não é versionada no repositório (binários gerados). Rode o notebook para recriá-la.

---

## Dashboard — dashboard.py

Aplicação **Streamlit** que lê os artefatos gerados pelo notebook e simula o monitoramento contínuo das estações, sem necessidade de re-treinar o modelo. Há duas versões:

- **`dashboard.py`** — versão simples, para o usuário final: 3 abas (mapa da cidade, acompanhar uma estação, resumo do dia), linguagem sem jargão, navegação por dia/hora e horizonte de previsão de 60 min (t+6).
- **`dashboard_tecnico_backup.py`** — versão técnica completa, com as 5 abas descritas a seguir (grafo, monitoramento, anomaly score, nós cegos e métricas do modelo).

As seções abaixo documentam a **versão técnica**. Na barra lateral dela, um slider de timestep (`t_sel`) simula o avanço do tempo no conjunto de teste de 2024.

---

### Aba 1 — Grafo

Mapa interativo de São Paulo com estações e arestas do grafo.

Cor dos nós por status no timestep selecionado:

| Cor | Significado |
|---|---|
| Verde | Estação operando normalmente |
| Vermelho | Alerta confirmado (Regra 1, qualquer variável) |
| Cinza | Nó cego (estimado via Feature Propagation) |

Tooltip ao passar o mouse exibe os valores previstos de `PLU`, `Temp`, `Umid.Rel.` e `Vel.VT`.

---

### Aba 2 — Monitoramento

Série temporal de uma estação e variável específica com previsão e alertas.

| Elemento | Descrição |
|---|---|
| Linha preta | Valor real observado em `t+1` |
| Linha azul tracejada | Previsão do modelo GNN |
| Banda azul claro | Intervalo P5–P95 histórico do mês atual |
| Marcadores vermelhos (×) | Instantes com alerta confirmado (Regra 1) |

Janela exibida: últimas 144 previsões (~24h a 10 minutos por leitura).

---

### Aba 3 — Anomaly Score

**Tabela de alertas:** todos os eventos com Regra 1 confirmada (nó, variável, instante, AS).

**Heatmap Estação × Variável:** contribuição por variável ao Anomaly Score no timestep selecionado:

$$\text{contrib}_{i,k} = |e_{i,k}| \cdot \sqrt{\Sigma^{-1}_{i,kk}}$$

Células mais vermelhas = erros mais atípicos em relação ao padrão histórico.

---

### Aba 4 — Nós Cegos

Comparação entre Feature Propagation + GNN (linha azul) e baseline de média simples dos vizinhos (linha laranja pontilhada), com MAE e RMSE entre os dois métodos.

---

### Aba 5 — Métricas do Modelo

- Resumo médio de MAE/RMSE por variável
- Comparação baseline vs FP + GraphSAGE por nó cego
- MAE/RMSE por nó (pluviosidade | temperatura), ordenados por RMSE crescente

---

### Arquitetura de dados (`artifacts/X_test.pkl`)

```
X_test.pkl
  ├── X_2024          [T, N, F]     features completas (FP aplicada)
  ├── X_2024_baseline [T, N, F]     baseline (média simples)
  ├── timestamps_2024               lista de timestamps
  ├── X_hat_mv        [T-1, N, F]   previsões do modelo (t+1)
  ├── X_real_next     [T-1, N, F]   valores reais em t+1
  ├── alert_confirmado[T-1, N, F]   máscara booleana de alertas confirmados
  ├── anomaly_scores  [T-1, N]      AS de Mahalanobis por nó e timestep
  ├── df_alertas                    DataFrame de alertas (Regra 1)
  ├── df_resultados_chuva           MAE/RMSE por nó (pluviosidade)
  ├── df_resultados_temp            MAE/RMSE por nó (temperatura)
  ├── resumo_pred                   resumo médio por variável
  └── df_metricas_cegos             comparação baseline vs FP por nó cego
```

---

### Abreviações

| Sigla | Significado |
|---|---|
| PLU | Pluviosidade (mm) |
| Temp | Temperatura / Sensação Térmica (°C) |
| Umid.Rel. | Umidade Relativa do Ar (%) |
| Vel.VT | Velocidade do Vento (m/s) |
| τ | Limiar de alerta (percentil 95 histórico) |
| AS | Anomaly Score (distância de Mahalanobis) |
| FP | Feature Propagation |
| GNN | Graph Neural Network |
| CGE | Centro de Gerenciamento de Emergências — Prefeitura de SP |
| INMET | Instituto Nacional de Meteorologia |
| IAG-USP | Instituto de Astronomia, Geofísica e Ciências Atmosféricas — USP |

---

## Datasets

### CGE — Centro de Gerenciamento de Emergências

Dados horários de 2009 a 2025 das estações meteorológicas da Prefeitura de SP.

> **Nota de formatação:** há uma vírgula extra entre as colunas `DATA` e `PLU(mm)` nos arquivos originais. É necessário um script para remover esse separador antes de usar os CSVs.

**Colunas:**

| Coluna | Significado |
|---|---|
| Posto | ID da estação no sistema CGE |
| DATA | Data e hora de captura |
| PLU(mm) | Pluviosidade acumulada |
| Vel.VT(m/s) | Velocidade do vento |
| Dir.VT(°) | Direção do vento |
| Temp(°C) | Temperatura do ar |
| Umid.Rel.(%) | Umidade relativa |
| Pressão(mb) | Pressão atmosférica |
| Bateria(V) | Tensão da bateria da estação |
| Rajada.VT(m/s) | Rajada máxima de vento |

**Obtenção:** solicitação à SIURB (Secretaria Municipal de Infraestrutura Urbana e Obras) via [e-SIC](https://esic.prefeitura.sp.gov.br/registrar_pedido_v2.aspx).

> **Dados e artefatos grandes:** o repositório versiona apenas os CSVs de 2023–2024.
> Os anos de treino **2016–2022** (`dados_cge_2016-2022.tar.gz`) e os artefatos que
> excedem 100 MB (`X_test.pkl`, `peryear_cache.pt`, `train_cache.pt`) estão publicados
> na **Release** do repositório no GitHub — baixe-os para reproduzir os resultados
> sem re-executar o notebook.

---

### INMET — Instituto Nacional de Meteorologia

Dados horários das estações automáticas. Possuímos dados de 2025.

Valores faltantes são identificados por `9999`, `Null` ou campo em branco.

**Colunas principais:**

| Coluna | Significado |
|---|---|
| Data / Hora UTC | Data e hora em UTC (não fuso local) |
| PRECIPITAÇÃO TOTAL, HORÁRIO (mm) | Pluviosidade acumulada na última hora |
| PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO (mB) | Pressão instantânea |
| PRESSÃO MAX./MIN. NA HORA ANT. (mB) | Pressão máxima e mínima da hora anterior |
| RADIACAO GLOBAL (Kj/m²) | Radiação solar na superfície |
| TEMPERATURA DO AR - BULBO SECO (°C) | Temperatura do ar |
| TEMPERATURA DO PONTO DE ORVALHO (°C) | Indicador de umidade |
| TEMPERATURA MAX./MIN. NA HORA ANT. (°C) | Extremos de temperatura na hora anterior |
| UMIDADE RELATIVA DO AR, HORARIA (%) | Umidade relativa instantânea |
| UMIDADE REL. MAX./MIN. NA HORA ANT. (%) | Extremos de umidade na hora anterior |
| VENTO, DIREÇÃO HORARIA (°) | Direção do vento (0–360°, 0 = norte) |
| VENTO, RAJADA MAXIMA (m/s) | Maior rajada na última hora |
| VENTO, VELOCIDADE HORARIA (m/s) | Velocidade média do vento |

**Obtenção:** dados públicos no portal do [INMET](http://portal.inmet.gov.br) — aba Dados Meteorológicos → [Banco de Dados Meteorológicos (BDMEP)](https://bdmep.inmet.gov.br/).

---

### IAG-USP — Instituto de Astronomia, Geofísica e Ciências Atmosféricas

Dados históricos de longa data: precipitação e temperatura desde 1933, pressão e umidade desde 1936, temperatura do solo desde 1962, velocidade do vento desde 1935.

**Colunas:**

| Coluna | Significado |
|---|---|
| Evap_arlivre | Evaporação em superfície livre (tanque Classe A) |
| Nuvens | Cobertura de nuvens (0–8 oitavos) |
| Prec | Precipitação (mm) |
| Press | Pressão atmosférica (hPa ou mb) |
| T ar | Temperatura do ar (°C) |
| T_solo | Temperatura do solo (°C) |
| UR | Umidade relativa do ar (%) |
| Vent | Velocidade e rajada máxima do vento |

**Obtenção:** [Formulário de Solicitação IAG-USP](https://www.estacao.iag.usp.br/sol_dados.php) — dúvidas: estacao.aca@iag.usp.br

---

## Plano de Melhorias

- [ ] Obtenção completa dos dados do **MackLEAPS** (site fora do ar)
- [ ] Tratamento avançado dos dados do INMET para integração completa
- [ ] Melhoria contínua nos nós cegos (IAG-USP, INMET e futuramente MackLEAPS)
- [ ] Testar outros modelos de previsão — o GraphSAGE prevê `t+i` com erro crescente, limitando o horizonte prático de previsão

---

## Referências

- **Hamilton, W. L., Ying, R., Leskovec, J.** "Inductive Representation Learning on Large Graphs." *NeurIPS 2017.* — Arquitetura GraphSAGE.

- **Rossi, E., Kenlay, H., Gorinova, M., Chamberlain, B., Dong, X., Bronstein, M.** "On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs With Missing Node Features." *ICLR 2022.* — Feature Propagation iterativa, energia de Dirichlet, interpolação harmônica.

- **Mahalanobis, P. C.** "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India, 1936.* — Distância de Mahalanobis para anomaly score multivariado.

- **Ortega, A., Frossard, P., Kovačević, J., Moura, J. M. F., Vandergheynst, P.** "Graph Signal Processing: Overview, Challenges, and Applications." *Proceedings of the IEEE, 2018.* — Fundamentos de processamento de sinais em grafos, Laplaciano e difusão.

- **Chung, F. R. K.** "Spectral Graph Theory." *AMS, 1997.* — Normalização espectral D⁻¹/²AD⁻¹/² e suas propriedades.

- **Akoglu, L., Tong, H., Koutra, D.** "Graph-based anomaly detection and description: a survey." *Data Mining and Knowledge Discovery, 2015.* — Taxonomia de anomalias (global, contextual, sazonal, tendência, intermetric).
