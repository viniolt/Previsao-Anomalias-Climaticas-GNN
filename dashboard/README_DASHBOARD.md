# dashboard.py — Documentação técnica

> **Atualização (24/07/2026):** `codigo/dashboard.py` agora é a **versão simples,
> voltada ao usuário final** (3 abas: mapa, estação, resumo do dia; linguagem sem
> jargão, navegação por dia/hora, horizonte corrigido para t+6 = 60 min). A versão
> técnica documentada abaixo (5 abas com anomaly score, nós cegos e métricas) foi
> preservada em `codigo/dashboard_tecnico_backup.py` — para restaurá-la, basta
> copiá-la de volta sobre `dashboard.py`. Uma cópia da versão técnica original
> também existe em `completo/dashboard.py`.

## O que este dashboard faz

`dashboard.py` é uma aplicação **Streamlit** de monitoramento climático em tempo real. Ele carrega os artefatos gerados pelo notebook `grafo-0.8.0-pyg.ipynb` e simula o monitoramento contínuo das estações meteorológicas de São Paulo, exibindo previsões, alertas e métricas do modelo GNN.

**Não é necessário rodar o modelo novamente.** O dashboard lê os resultados já salvos e permite navegar interativamente pelo histórico de previsões e alertas.

---

## Pré-requisito: rodar o notebook

O dashboard depende da pasta `artifacts/` gerada pelo notebook. Se ela não existir, o dashboard exibe uma mensagem de erro e para.

Arquivos esperados em `artifacts/`:

| Arquivo | Gerado na célula | Conteúdo |
|---|---|---|
| `graph_data.pkl` | 45 | Estrutura do grafo (nós, arestas, coordenadas, nomes, nós cegos) |
| `tau.pkl` | 45 | Limiares τ `[12, N, F]` — percentil 95 por mês/estação/variável |
| `sigma_inv.pkl` | 45 | Inversas de covariância `[N, F, F]` para distância de Mahalanobis |
| `X_test.pkl` | 45 | Tensores de teste, previsões `X_hat_mv`, alertas, métricas |
| `model_plu.pt` | 45 | Pesos do modelo de pluviosidade (não carregado diretamente pelo dashboard) |
| `model_temp.pt` | 45 | Pesos do modelo de temperatura (não carregado diretamente pelo dashboard) |

---

## Como iniciar

```bash
/home/viniz0r/Documentos/IT_novo/.venv/bin/streamlit run dashboard/dashboard.py
```

Acesse no navegador: **http://localhost:8501**

---

## Interface: visão geral

A barra lateral contém um **slider de timestep** (`t_sel`) que simula o avanço do tempo. Cada posição do slider representa um instante de previsão `t+1` no conjunto de teste de 2024. Mover o slider atualiza todas as abas simultaneamente.

A barra lateral também exibe:
- O instante de tempo correspondente ao timestep selecionado
- O total de estações (`N_NODES`)
- Os índices dos nós cegos (`blind_nodes`)

---

## Aba 1 — Grafo

**O que mostra:** mapa interativo de São Paulo com todas as estações meteorológicas e as arestas do grafo de conectividade.

**Como funciona:**
- Cada nó é exibido em uma de três cores:
  - **Verde** — estação operando normalmente no timestep selecionado
  - **Vermelho** — estação com alerta ativo (Regra 1 confirmada em qualquer variável)
  - **Cinza** — nó cego (sem CSV próprio; valor estimado por Feature Propagation)
- As **arestas** (conexões entre estações) são desenhadas como linhas cinza-claro
- Ao passar o mouse sobre um nó, um tooltip exibe os valores previstos de cada variável (`PLU`, `Temp`, `Umid.Rel.`, `Vel.VT`) no instante selecionado
- O mapa usa o estilo `carto-positron` via Plotly/Mapbox e é centralizado automaticamente na média das coordenadas das estações

**Dado carregado:** `graph_data.pkl` (coordenadas e arestas) + `alert_confirmado[t_sel]` (status de alerta por nó).

---

## Aba 2 — Monitoramento

**O que mostra:** série temporal de uma estação e variável específica, com a previsão do modelo e marcadores de alerta.

**Controles:**
- Seletor de **estação** (qualquer nó do grafo)
- Seletor de **variável** (`PLU`, `Temp`, `Umid.Rel.`, `Vel.VT`)

**Conteúdo do gráfico:**
- **Linha preta** — valor real observado no instante `t+1` (`X_real_next`)
- **Linha azul tracejada** — previsão do modelo GNN (`X_hat_mv`)
- **Banda azul claro** — intervalo P5–P95 do mês atual, construído a partir de:
  - P95 = limiar τ calculado no notebook (percentil 95 histórico por mês/estação/variável)
  - P5 = percentil 5 calculado sobre `X_2024` para o nó e variável selecionados
- **Marcadores vermelhos (×)** — instantes com alerta **confirmado** pela Regra 1 (3 timesteps consecutivos acima do limiar)

A janela exibida compreende as últimas **144 previsões** antes do timestep selecionado (~24h a 10 minutos por leitura).

**Fundamento teórico — Regra 1 (limiar previsto):**

$$\tau_{i,m,k} = \text{P}_{95}\bigl(x_{i,k} \mid \text{mês} = m\bigr)$$

Alerta confirmado quando $\hat{x}_{i,k,t} > \tau_{i,m,k}$ por $W_{\text{alert}} = 3$ timesteps consecutivos. A janela anti-ruído evita falsos positivos por spikes isolados de sensor.

---

## Aba 3 — Anomaly Score

**O que mostra:** tabela de alertas confirmados e heatmap do Anomaly Score multivariado.

### Tabela de alertas

Lista todos os eventos em que a Regra 1 foi confirmada (3 timesteps consecutivos acima do limiar P95), com colunas de nó, variável, instante e valor do Anomaly Score (AS).

### Heatmap Estação × Variável

Para o timestep selecionado, calcula a **contribuição por variável** ao Anomaly Score de cada estação:

$$\text{contrib}_{i,k} = |e_{i,k}| \cdot \sqrt{\Sigma^{-1}_{i,kk}}$$

onde $e_{i,k} = x_{i,k,t+1} - \hat{x}_{i,k,t+1}$ é o erro de previsão e $\Sigma^{-1}_{i,kk}$ é o elemento diagonal da inversa de covariância da estação $i$ para a variável $k$.

Células mais vermelhas indicam variáveis e estações onde o erro de previsão é mais atípico em relação ao padrão histórico.

**Fundamento teórico — Regra 2 (Mahalanobis):**

O Anomaly Score total por nó é:

$$AS_{i,t} = \sqrt{e_{i,t}^\top \, \Sigma_i^{-1} \, e_{i,t}}$$

A distância de Mahalanobis leva em conta a **covariância entre variáveis**: se pluviosidade e umidade sobem juntas, isso é esperado e não gera alerta. Mas se temperatura cai enquanto velocidade do vento sobe de forma incomum para aquela estação e mês, o AS será alto mesmo que cada variável individualmente esteja dentro do limiar.

---

## Aba 4 — Nós Cegos

**O que mostra:** comparação entre os dois métodos de estimativa das estações sem dados próprios.

**Nós cegos:**
- `33` — IAG-USP (Parque Estadual das Fontes do Ipiranga)
- `34` — INMET automático
- `35` — INMET convencional

**Controle:** seletor de variável (`PLU`, `Temp`, etc.)

Para cada nó cego, o gráfico exibe:
- **Linha azul** — estimativa por **Feature Propagation + GNN** (40 iterações de difusão + inferência da GraphSAGE)
- **Linha laranja pontilhada** — estimativa **baseline** (média simples dos vizinhos, 1 iteração sem restauração de âncoras)

Abaixo do gráfico, dois indicadores (`st.metric`) exibem:
- **MAE** (Erro Absoluto Médio) entre os dois métodos ao longo de todo o período de teste
- **RMSE** (Raiz do Erro Quadrático Médio) entre os dois métodos

**O que esperar:** as curvas FP+GNN tendem a ser mais suaves e a capturar melhor padrões espaciais de longo alcance, enquanto o baseline depende apenas dos vizinhos imediatos.

**Fundamento teórico:** ver Rossi et al. (ICLR 2022) — a Feature Propagation converge para a interpolação harmônica da energia de Dirichlet, que é a solução ótima para estimar features em nós com dados faltantes dado o grafo de conectividade.

---

## Aba 5 — Métricas do Modelo

**O que mostra:** tabelas de desempenho do modelo no conjunto de teste de 2024.

### Resumo médio por variável

Média de MAE e RMSE sobre todos os nós com dados reais, calculada separadamente para pluviosidade e temperatura.

### Nós cegos: baseline vs FP + GraphSAGE

Tabela comparando MAE e RMSE entre os dois métodos de estimativa, segmentada por nó cego e variável.

### MAE/RMSE por nó

Duas tabelas lado a lado (pluviosidade | temperatura) com o desempenho do modelo GNN em cada estação, ordenadas por RMSE crescente. Permite identificar quais estações o modelo prevê melhor e onde há mais erro.

---

## Arquitetura de dados

```
artifacts/X_test.pkl contém:
  ├── X_2024          [T, N, F]     tensor de features (FP aplicada)
  ├── X_2024_baseline [T, N, F]     tensor baseline (média simples)
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

## Variáveis e abreviações

| Abreviação | Significado |
|---|---|
| PLU | Pluviosidade (mm) |
| Temp | Temperatura / Sensação Térmica (°C) |
| Umid.Rel. | Umidade Relativa do Ar (%) |
| Vel.VT | Velocidade do Vento (km/h ou m/s) |
| τ | Limiar de alerta (percentil 95 histórico por mês/estação/variável) |
| AS | Anomaly Score (distância de Mahalanobis sobre erro de previsão) |
| FP | Feature Propagation |
| GNN | Graph Neural Network |
| CGE | Centro de Gerenciamento de Emergências — Prefeitura de SP |
| INMET | Instituto Nacional de Meteorologia |
| IAG-USP | Instituto de Astronomia, Geofísica e Ciências Atmosféricas — USP |

---

## Referências

- **Hamilton, W. L., Ying, R., Leskovec, J.** "Inductive Representation Learning on Large Graphs." *NeurIPS 2017.* — Arquitetura GraphSAGE usada no modelo de previsão.

- **Rossi, E., Kenlay, H., Gorinova, M., Chamberlain, B., Dong, X., Bronstein, M.** "On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs With Missing Node Features." *ICLR 2022.* — Feature Propagation iterativa para estimativa dos nós cegos.

- **Mahalanobis, P. C.** "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India, 1936.* — Base teórica do Anomaly Score multivariado (Regra 2).

- **Ortega, A., Frossard, P., Kovačević, J., Moura, J. M. F., Vandergheynst, P.** "Graph Signal Processing: Overview, Challenges, and Applications." *Proceedings of the IEEE, 2018.* — Fundamentos de difusão de sinais em grafos e normalização espectral.

- **Akoglu, L., Tong, H., Koutra, D.** "Graph-based anomaly detection and description: a survey." *Data Mining and Knowledge Discovery, 2015.* — Taxonomia de anomalias (global, contextual, sazonal, tendência, intermetric).
