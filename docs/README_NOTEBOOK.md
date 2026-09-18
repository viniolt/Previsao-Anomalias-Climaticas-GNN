# grafo-0.8.0-pyg.ipynb — Documentação técnica

> **AVISO DE VERSÃO (jul/2026):** este documento descreve uma versão anterior do
> notebook e diverge do pipeline atual em quatro pontos. O estado vigente
> (o mesmo do artigo) é: **treino 2016–2022, validação retida 2023, teste 2024**
> (não "treino 2023"); grade de 10 min com horizonte **t+6 = 60 min** (não t+1);
> grafo **`003.graphml`**; e o Feature Propagation usa a normalização
> **estocástica por linha D⁻¹A** (`edge_weight`), NÃO a simétrica
> `edge_weight_sym` — a simétrica não soma 1 nos nós de destino e distorce a
> reconstrução dos nós cegos (ver Seção III-A do artigo e
> `reproduz_artigo/pesos.py`). Onde este texto contradisser o notebook ou o
> artigo, valem o notebook e o artigo.

## O que este notebook faz

Este notebook implementa um sistema de **previsão e detecção de anomalias climáticas** para a rede de estações meteorológicas de São Paulo, usando uma **Graph Neural Network (GNN)** do tipo GraphSAGE sobre dados de pluviosidade e temperatura.

A versão 0.8.0 introduz três melhorias em relação à 0.7.5:

1. **Normalização das arestas por nó de destino** (D⁻¹A)
2. **Feature Propagation iterativa** nos nós sem dados (Rossi et al., ICLR 2022)
3. **Detecção de anomalias** com limiar de percentil + distância de Mahalanobis

---

## Pré-requisitos

| Dependência | Versão testada |
|---|---|
| Python | 3.12 |
| PyTorch | 2.11.0+cu130 |
| PyTorch Geometric (PyG) | 2.7.0 |
| NetworkX | qualquer |
| pandas, numpy, matplotlib | qualquer |

**Ambiente recomendado:** `/home/viniz0r/Documentos/IT_novo/.venv`

**Dados necessários:**
- `estacoes_meteorologicas/meteorologicas_2023/` — CSVs das estações CGE (treino)
- `estacoes_meteorologicas/meteorologicas_2024/` — CSVs das estações CGE (teste)
- `001.graphml` — arquivo de grafo com arestas, pesos e coordenadas geográficas

Após rodar o notebook por completo, a pasta `artifacts/` é criada com os artefatos necessários para o dashboard.

---

## Estrutura do notebook (célula a célula)

### Células 1–2: Importações e seleção de dispositivo

Importa todas as bibliotecas e define `selecionar_dispositivo()`, que escolhe automaticamente entre CUDA, MPS (Apple Silicon) e CPU.

### Células 3–5: Carregamento do grafo e funções geográficas

Carrega o grafo de estações a partir de `001.graphml` usando NetworkX. Define funções auxiliares:
- `dms_to_decimal()` — converte coordenadas de graus/minutos/segundos para decimal
- `haversine()` — calcula distância em km entre dois pontos geográficos

### Células 6–9: Leitura dos CSVs das estações

Define os caminhos `PATH_2023` e `PATH_2024` e implementa toda a lógica de leitura robusta dos CSVs das estações CGE:
- Detecta separador (`,` ou `;`) automaticamente
- Corrige cabeçalhos malformados
- Normaliza nomes de estações (remove acentos, maiúsculas, stopwords)
- Mapeia cada arquivo CSV ao nó correspondente no grafo

### Célula 10–11: Montagem do tensor `[T, N, F]`

Constrói o tensor principal de dados com forma `[timesteps, nós, atributos]` onde:
- `T` = número de timesteps (cada 10 minutos, ~2023+2024)
- `N` = número de nós no grafo (estações)
- `F` = 4 atributos: `PLU` (pluviosidade), `Temp` (temperatura), `Umid.Rel.` (umidade relativa), `Vel.VT` (velocidade do vento)

**Nós cegos** (`33` = IAG-USP, `34` = INMET automático, `35` = INMET convencional): não possuem CSV próprio. São inicializados com zeros e depois estimados via Feature Propagation.

---

## Mudança 1 — Normalização das arestas por nó de destino

### Problema da versão anterior

Na versão 0.7.5, os pesos das arestas eram normalizados por min-max global:

```
w̃ = (w - w_min) / (w_max - w_min)
```

Se uma única aresta tinha peso muito alto (estações muito próximas), ela comprimia todos os outros pesos para perto de zero, dominando a agregação.

### Correção implementada (célula 12)

Normalização por **nó de destino**, garantindo que a soma dos pesos das arestas que chegam a cada nó seja exatamente 1:

$$\tilde{w}_{ij} = \frac{w_{ij}}{\sum_{k \in \mathcal{N}(j)} w_{kj}}$$

Isso equivale à normalização **D⁻¹A** (multiplicação pela inversa do grau, por linha do nó destino). Também é definida a versão simétrica **D⁻¹/²AD⁻¹/²** para uso opcional:

$$\tilde{w}_{ij}^{\text{sym}} = \frac{w_{ij}}{\sqrt{d_i \cdot d_j}}$$

O objeto `Data` do PyG (`g_pyg`) é construído com ambas as versões (`edge_weight` e `edge_weight_sym`).

---

## Mudança 2 — Feature Propagation nos nós cegos

### Motivação

Estações sem CSV (nós cegos) precisam ter suas features estimadas a partir dos vizinhos. A abordagem anterior usava uma simples média dos vizinhos diretos — equivalente a uma única iteração de difusão sem restauração das âncoras.

### Fundamento teórico

Baseado em **Rossi et al., "On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs With Missing Node Features", ICLR 2022**.

O algoritmo minimiza a **energia de Dirichlet** do sinal no grafo:

$$\ell(x, G) = \frac{1}{2} x^\top \Delta x = \frac{1}{2} \sum_{ij} \tilde{a}_{ij} (x_i - x_j)^2$$

onde $\Delta = I - \tilde{A}$ é o Laplaciano normalizado e $\tilde{A} = D^{-1/2} A D^{-1/2}$ é a adjacência normalizada simetricamente.

A solução é obtida iterativamente:
1. **Difusão:** $X \leftarrow \tilde{A} X$ — cada nó recebe a média ponderada dos vizinhos
2. **Restauração:** $X[\text{âncoras}] \leftarrow X_{\text{original}}[\text{âncoras}]$ — nós com dados reais são fixados

Após 40 iterações, os nós cegos convergem para a interpolação harmônica ótima.

### Implementação vetorizada (célula 15)

Para eficiência, todos os `T` timesteps são processados **em paralelo** sobre o tensor `[T, N, F]`:

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
        X_new.index_add_(1, dst, msgs)       # agrega por nó destino
        X_new[:, known_mask, :] = X_known   # restaura âncoras
        X = X_new
    return X
```

---

## Modelo GraphSAGE (células 18–21)

A arquitetura usada é `MeteorologiaGNN`, baseada em **GraphSAGE** (Hamilton et al., 2017):

```
Entrada: x ∈ ℝ^{N×4}
  └─ WeightedSAGEConv(4 → 32)  + ReLU
  └─ WeightedSAGEConv(32 → 32) + ReLU
  └─ Linear(32 → 1)
Saída: ŷ ∈ ℝ^N  (um escalar por nó)
```

Cada camada `WeightedSAGEConv` implementa:

$$h'_i = W_{\text{self}} \cdot x_i + W_{\text{neigh}} \cdot \frac{\sum_{j \in \mathcal{N}(i)} w_{ij} \cdot x_j}{\sum_{j} w_{ij}}$$

### Treino (células 22–25)

- Dois modelos separados: **Pluviosidade** e **Temperatura**
- Dados de 2023 como treino, 2024 como teste
- Loss: MSE apenas sobre os nós **com dados reais** (nós cegos excluídos do gradiente via máscara `IDX_TREINO`)
- Otimizador: Adam, lr=0.001, 50 épocas máximo, early stopping com paciência=2

---

## Mudança 3 — Detecção de anomalias (célula 41)

### Regra 1 — Limiar meteorológico previsto

Para cada estação $i$, mês $m$ e variável $k$, pré-computa o **percentil 95** histórico:

$$\tau_{i,m,k} = \text{P}_{95}\bigl(x_{i,k} \mid \text{mês} = m\bigr)$$

Alerta disparado quando a previsão supera o limiar: $\hat{x}_{i,k} > \tau_{i,m,k}$

**Janela anti-ruído** ($W_{\text{alert}} = 3$): o alerta só é **confirmado** após 3 timesteps consecutivos acima do limiar. Isso evita falsos positivos causados por spikes de sensor.

### Regra 2 — Anomaly Score multivariado (Mahalanobis)

Para cada nó $i$ e timestep $t$, define o erro de previsão:

$$e_{i,t} = x_{i,t+h} - \hat{x}_{i,t+h} \in \mathbb{R}^F$$

O **Anomaly Score** é a distância de Mahalanobis sobre o vetor de erro:

$$AS_{i,t} = \sqrt{e_{i,t}^\top \, \Sigma_i^{-1} \, e_{i,t}}$$

onde $\Sigma_i^{-1}$ é a inversa da covariância estimada nos erros de treino da estação $i$. Valores altos indicam que o padrão multivariado do erro é atipico — mesmo que nenhuma variável individual tenha ultrapassado seu limiar.

---

## Métricas de avaliação (células 42–43)

Para os nós cegos, compara dois métodos de estimativa:

| Método | Descrição |
|---|---|
| Baseline | Média simples dos vizinhos diretos (1 iteração, sem restauração) |
| FP + GraphSAGE | Feature Propagation (40 iter.) + inferência da GNN |

Métricas: **MAE** e **RMSE** por nó e por variável.

---

## Exportação de artefatos (célula 45)

Ao final, tudo necessário para o dashboard é salvo em `artifacts/`:

| Arquivo | Conteúdo |
|---|---|
| `model_plu.pt` | Pesos do modelo de pluviosidade |
| `model_temp.pt` | Pesos do modelo de temperatura |
| `tau.pkl` | Limiares τ com forma `[12, N, F]` (percentil 95 por mês) |
| `sigma_inv.pkl` | Inversas de covariância `[N, F, F]` para Mahalanobis |
| `graph_data.pkl` | Estrutura do grafo: nós, arestas, coordenadas, nomes |
| `X_test.pkl` | Tensores de teste, previsões, alertas e métricas |

---

## Como executar

1. Ativar o ambiente virtual:
   ```bash
   source /home/viniz0r/Documentos/IT_novo/.venv/bin/activate
   ```

2. Abrir o notebook no VS Code ou JupyterLab:
   ```bash
   jupyter notebook grafo-0.8.0-pyg.ipynb
   ```

3. Executar todas as células em ordem (`Run All`). O treino leva aproximadamente 10–15 minutos com GPU.

4. Ao final, verificar que `artifacts/` contém os 6 arquivos listados acima.

---

## Referências

- **Hamilton, W. L., Ying, R., Leskovec, J.** "Inductive Representation Learning on Large Graphs." *NeurIPS 2017.* — Arquitetura GraphSAGE.

- **Rossi, E., Kenlay, H., Gorinova, M., Chamberlain, B., Dong, X., Bronstein, M.** "On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs With Missing Node Features." *ICLR 2022.* — Feature Propagation iterativa, energia de Dirichlet, interpolação harmônica.

- **Mahalanobis, P. C.** "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India, 1936.* — Distância de Mahalanobis para anomaly score multivariado.

- **Ortega, A., Frossard, P., Kovačević, J., Moura, J. M. F., Vandergheynst, P.** "Graph Signal Processing: Overview, Challenges, and Applications." *Proceedings of the IEEE, 2018.* — Fundamentos de processamento de sinais em grafos, Laplaciano e difusão.

- **Chung, F. R. K.** "Spectral Graph Theory." *AMS, 1997.* — Normalização espectral D⁻¹/²AD⁻¹/² e suas propriedades.
