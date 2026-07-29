# Roteiro de Apresentação

Dois roteiros neste arquivo, para contextos diferentes:

- **Parte 1 — Apresentação de IC (v0.8.0):** para reuniões com o orientador ou banca de IC. Foco no progresso e nas mudanças técnicas da versão.
- **Parte 2 — Apresentação do Artigo (ENIAC/BRACIS):** para conferência científica. Foco na contribuição completa como trabalho publicável.

---

# Parte 1 — Apresentação de IC (v0.8.0)
**Tempo estimado: 12–15 minutos**

---

## Slide 1 — Abertura (1 min)

> "Hoje eu vou apresentar a versão 0.8.0 do nosso sistema de previsão de anomalias climáticas para São Paulo. O sistema usa uma Graph Neural Network — uma rede neural que aprende sobre grafos — aplicada à rede de estações meteorológicas do CGE, da USP e do INMET."

**Ponto de partida:**
- A versão anterior (0.7.5) já funcionava: carregava dados, construía o grafo, treinava o modelo e fazia previsões
- Mas tinha três problemas que precisavam ser corrigidos
- Esta versão resolve os três e adiciona um dashboard interativo

---

## Slide 2 — Contexto: como o sistema funciona (2 min)

> "Antes de entrar nas mudanças, vale revisitar rapidamente a estrutura do sistema."

**O grafo:**
- Cada **nó** é uma estação meteorológica (36 no total)
- Cada **aresta** conecta estações próximas geograficamente, com peso inversamente proporcional à distância (Haversine)
- Os dados de entrada são 4 variáveis por nó a cada 10 minutos: **pluviosidade, temperatura, umidade relativa e velocidade do vento**

**O modelo:**
- Uma **GraphSAGE**: a cada camada, cada estação agrega informação dos seus vizinhos ponderada pelo peso da aresta
- Dois modelos separados: um para prever pluviosidade, outro para temperatura
- Treino em 2023, teste em 2024

**O problema:**
- Três estações — IAG-USP (nó 33), INMET automático (34) e INMET convencional (35) — **não têm CSV próprio**
- Chamamos elas de **nós cegos**: o modelo precisa estimá-las a partir dos vizinhos

---

## Slide 3 — Problema 1: os pesos das arestas estavam errados (2 min)

> "A primeira mudança resolve um problema sutil mas importante na forma como as conexões entre estações eram ponderadas."

**O que a versão 0.7.5 fazia:**

Normalização min-max global sobre todos os pesos:

$$\tilde{w} = \frac{w - w_{\min}}{w_{\max} - w_{\min}}$$

**Por que isso é um problema:**
- Se duas estações são muito próximas, o peso daquela aresta domina
- Todas as outras arestas ficam comprimidas perto de zero
- A rede neural "escuta" quase só aquele vizinho mais próximo — mesmo que outros vizinhos tragam informação relevante

**A correção — normalização por nó de destino (D⁻¹A):**

$$\tilde{w}_{ij} = \frac{w_{ij}}{\displaystyle\sum_{k \in \mathcal{N}(j)} w_{kj}}$$

> "Agora, para cada estação, a soma dos pesos dos seus vizinhos é sempre 1. Cada vizinho contribui proporcionalmente à sua proximidade, sem que nenhum domine de forma artificial."

Isso equivale a multiplicar a matriz de adjacência pela inversa da matriz de grau — o que na literatura é chamado de normalização **D⁻¹A**.

---

## Slide 4 — Problema 2: os nós cegos eram mal estimados (3 min)

> "A segunda mudança é a mais relevante do ponto de vista científico."

**O que a versão 0.7.5 fazia:**
- Preenchia os nós cegos com a **média simples dos vizinhos diretos**
- Isso é equivalente a fazer uma única iteração de difusão no grafo — muito superficial

**O problema:**
- A média simples não considera a estrutura global do grafo
- Vizinhos distantes que têm informação relevante são ignorados
- O resultado é uma estimativa "crua" que não respeita os padrões espaciais do fenômeno

**A solução: Feature Propagation (Rossi et al., ICLR 2022)**

> "Em 2022, Rossi e colaboradores mostraram que um algoritmo iterativo muito simples — chamado Feature Propagation — consegue estimar features faltantes em grafos com qualidade surpreendente, superando métodos muito mais complexos."

O algoritmo minimiza a **energia de Dirichlet** do sinal no grafo:

$$\ell(x, G) = \frac{1}{2} \sum_{ij} \tilde{a}_{ij}(x_i - x_j)^2$$

> "Minimizar essa energia significa encontrar valores para os nós cegos que sejam o mais parecido possível com os seus vizinhos — a interpolação mais suave possível."

**Como funciona na prática — duas etapas por iteração:**

1. **Difusão:** cada nó recebe a média ponderada dos seus vizinhos
   $$X \leftarrow \tilde{A} X$$

2. **Restauração:** os nós com dados reais (âncoras) voltam ao valor original
   $$X[\text{âncoras}] \leftarrow X_{\text{real}}[\text{âncoras}]$$

Repetido **40 vezes**, os nós cegos convergem para a interpolação harmônica ótima — a solução exata do problema de minimização.

**Detalhe de implementação:**
- O dataset tem ~50.000 timesteps
- Rodar FP para cada timestep separadamente levaria horas
- Solução: processar todos os timesteps **em paralelo** sobre o tensor `[T, N, F]` usando operações vetorizadas do PyTorch

---

## Slide 5 — Problema 3: não havia detecção de anomalias (3 min)

> "A terceira mudança é o que transforma o sistema de 'previsão' em 'monitoramento'. Implementamos duas regras complementares para identificar comportamentos anômalos."

### Regra 1 — Limiar meteorológico previsto

> "A primeira regra é a mais intuitiva: comparar a previsão do modelo com um limiar histórico."

Para cada combinação de estação, mês e variável, calculamos o **percentil 95** histórico:

$$\tau_{i,m,k} = \text{P}_{95}\bigl(x_{i,k} \mid \text{mês} = m\bigr)$$

Alerta quando a **previsão** supera esse limiar: $\hat{x}_{i,k} > \tau_{i,m,k}$

**Vantagem:** é um alerta **antecipado** — dispara antes de o evento acontecer, com base na previsão da GNN.

**Problema:** sensores têm spikes esporádicos que causariam falsos positivos.

**Solução — janela anti-ruído ($W_{\text{alert}} = 3$):**
> "O alerta só é confirmado se o modelo prevê o valor acima do limiar por 3 timesteps consecutivos — ou seja, por 30 minutos seguidos. Ruído de sensor some depois de um ou dois pontos. Um evento real persiste."

### Regra 2 — Anomaly Score multivariado (Mahalanobis)

> "A segunda regra captura algo que a primeira não consegue: padrões multivariados incomuns."

O erro de previsão para cada estação é um vetor com F variáveis:

$$e_{i,t} = x_{i,t+1} - \hat{x}_{i,t+1} \in \mathbb{R}^F$$

O **Anomaly Score** é a distância de Mahalanobis sobre esse vetor:

$$AS_{i,t} = \sqrt{e_{i,t}^\top \, \Sigma_i^{-1} \, e_{i,t}}$$

> "A distância de Mahalanobis considera a correlação entre variáveis. Se temperatura e umidade sobem juntas, isso é esperado e não gera alerta. Mas se o modelo prevê que vai chover muito enquanto o vento cai — algo raro naquela estação naquele mês — o AS será alto, mesmo que cada variável individualmente esteja dentro do limiar."

$\Sigma_i^{-1}$ é a inversa da covariância estimada nos erros de treino da estação $i$.

---

## Slide 6 — Dashboard de monitoramento (2 min)

> "Todo esse sistema agora tem uma interface interativa construída com Streamlit e Plotly."

**Como funciona:**
- Carrega os artefatos salvos pelo notebook (modelos, limiares, previsões, alertas)
- Um slider na barra lateral permite navegar pelo tempo, simulando o monitoramento em tempo real

**5 abas:**

| Aba | O que mostra |
|---|---|
| **Grafo** | Mapa de São Paulo com estações: verde = normal, vermelho = alerta, cinza = nó cego |
| **Monitoramento** | Série temporal de qualquer estação/variável com previsão + banda P5-P95 + marcadores de alerta |
| **Anomaly Score** | Tabela de alertas confirmados + heatmap estação × variável das contribuições ao AS |
| **Nós Cegos** | Comparação FP+GNN vs baseline (média simples) para os nós 33, 34 e 35 |
| **Métricas** | MAE e RMSE por nó e variável — pluviosidade e temperatura |

---

## Slide 7 — Resumo das mudanças (1 min)

| # | Mudança | Antes | Depois |
|---|---|---|---|
| 1 | Normalização das arestas | Min-max global | D⁻¹A por nó de destino |
| 2 | Estimativa dos nós cegos | Média simples (1 iter.) | Feature Propagation (40 iter., Rossi 2022) |
| 3 | Detecção de anomalias | Nenhuma | Regra 1 (P95 + anti-ruído) + Regra 2 (Mahalanobis) |
| + | Interface | Gráficos estáticos | Dashboard Streamlit interativo |

---

## Slide 8 — Referências (30 seg)

- **Hamilton et al., NeurIPS 2017** — GraphSAGE
- **Rossi et al., ICLR 2022** — Feature Propagation e energia de Dirichlet
- **Mahalanobis, 1936** — Distância de Mahalanobis para anomaly score multivariado
- **Ortega et al., IEEE 2018** — Graph Signal Processing e normalização espectral
- **Akoglu et al., DAMI 2015** — Taxonomia de anomalias em grafos

---

## Dicas para a apresentação de IC

- **Slides 1–2:** contexto geral — não assuma que todos lembram do sistema anterior
- **Slides 3–4:** são os mais densos; use os gráficos do notebook como apoio visual
- **Slide 5:** a analogia da temperatura + umidade ajuda muito a explicar o Mahalanobis para não-especialistas
- **Slide 6:** se possível, abrir o dashboard ao vivo e navegar o slider enquanto fala
- **Slide 7:** feche com a tabela — ela sintetiza o progresso de forma clara

---
---

# Parte 2 — Apresentação do Artigo (ENIAC / BRACIS)
**Tempo estimado: 10–12 minutos + 3 min de perguntas**

> Diferença de contexto: aqui você não está mostrando progresso incremental — está apresentando um trabalho completo e autossuficiente para revisores e pesquisadores que nunca viram o projeto. Cada slide precisa justificar a existência do trabalho, não apenas descrever o que foi feito.

---

## Slide 1 — Título e contexto (30 seg)

> "O trabalho que apresento hoje propõe um sistema de detecção de anomalias climáticas para São Paulo usando Graph Neural Networks. O diferencial está em como tratamos dois problemas simultâneos: estações sem dados próprios dentro da rede, e a natureza multivariada e espacialmente correlacionada dos eventos extremos."

**No slide:** título, autores, instituição, venue.

---

## Slide 2 — Motivação: por que isso importa (1 min)

> "São Paulo é a maior cidade da América do Sul e convive com eventos climáticos extremos frequentes: alagamentos, tempestades, ondas de calor. Detectar esses eventos com antecedência — antes que aconteçam — pode reduzir danos materiais e salvar vidas."

**Três pontos no slide:**
- Eventos extremos têm assinatura multivariada: chuva + vento + queda de pressão juntos
- Redes de monitoramento têm lacunas — algumas estações não transmitem dados
- Modelos tradicionais de série temporal ignoram a dependência espacial entre estações

> "Nossa proposta aborda os três pontos com um único sistema."

---

## Slide 3 — Contribuições (30 seg)

> "O trabalho tem três contribuições técnicas principais."

**No slide, três bullets curtos:**

1. **Interpolação harmônica** de estações sem dados via Feature Propagation iterativa sobre o grafo
2. **Previsão espaço-temporal** com GraphSAGE ponderado por distância geográfica
3. **Detecção dupla** de anomalias: limiar de percentil histórico + distância de Mahalanobis multivariada

---

## Slide 4 — Trabalhos relacionados (1 min)

> "Situando o trabalho na literatura."

**Três blocos rápidos:**

- **Detecção de anomalias em séries temporais** (Darban et al., 2022): survey abrangente — modelos de previsão usam erro de previsão como anomaly score; modelos de reconstrução usam erro de reconstrução. Nosso trabalho segue a abordagem de previsão.
- **GNNs para dados espaço-temporais** (Hamilton et al., 2017; Gilmer et al., 2017): GraphSAGE propõe agregação indutiva — aprende com a estrutura local do grafo, generalizável a nós não vistos no treino.
- **Features faltantes em grafos** (Rossi et al., 2022): Feature Propagation supera métodos complexos com um algoritmo iterativo simples baseado em minimização de energia de Dirichlet.

> "A combinação de Feature Propagation para nós sem dados seguida de GNN para previsão e detecção não foi explorada no contexto de monitoramento climático urbano."

---

## Slide 5 — Metodologia: o grafo e os dados (1.5 min)

> "O sistema modela a rede de estações como um grafo onde a topologia captura a dependência espacial."

**Construção do grafo:**
- **Nós:** 36 estações meteorológicas do CGE, INMET e IAG-USP
- **Arestas:** pares de estações conectadas por proximidade geográfica
- **Pesos:** inverso da distância Haversine, normalizado por nó de destino (D⁻¹A)

> "A normalização por nó de destino é necessária porque duas estações estavam a apenas 0,1 km de distância — sem correção, aquela única aresta concentrava toda a influência na agregação."

**Dados:**
- 4 variáveis por nó, a cada 10 minutos: PLU, Temp, Umid.Rel., Vel.VT
- Treino: 2023 · Teste: 2024
- 3 nós sem dados próprios — chamados de **nós cegos**

---

## Slide 6 — Metodologia: Feature Propagation (1.5 min)

> "Para os três nós sem dados, aplicamos Feature Propagation antes de alimentar a GNN."

**A hipótese:** estações geograficamente próximas tendem a ter condições climáticas parecidas — homofilia de features.

**O objetivo formal:** encontrar valores para os nós cegos que minimizem a energia de Dirichlet:

$$\mathcal{E}(X, G) = \frac{1}{2}\sum_{(i,j)\in E} \tilde{a}_{ij}\|x_i - x_j\|^2$$

**O algoritmo (2 passos por iteração, 40 iterações):**

1. Difusão: $X^{(r)} \leftarrow \tilde{A}\,X^{(r-1)}$
2. Restauração das âncoras: $X^{(r)}_{\mathcal{K}} \leftarrow X^{(0)}_{\mathcal{K}}$

> "Após a convergência, os nós cegos carregam a interpolação harmônica ótima — a estimativa mais coerente com a estrutura do grafo. Toda a operação é vetorizada sobre os ~50.000 timesteps em paralelo na GPU."

---

## Slide 7 — Metodologia: GraphSAGE e detecção (1.5 min)

**GraphSAGE:**
- Dois modelos separados (PLU e Temp), arquitetura 4→32→1
- Cada camada agrega vizinhos ponderados pelos pesos geográficos:

$$h'_i = W_{\text{self}}\,x_i + W_{\text{neigh}}\,\frac{\sum_{j\in\mathcal{N}(i)}\tilde{w}_{ij}\,x_j}{\sum_j \tilde{w}_{ij}}$$

- Horizonte h = 6 passos (~60 min) · Loss MSE só nos nós com dados reais

**Detecção dupla:**

| Regra | Critério | Captura |
|---|---|---|
| R1 — Limiar | $\hat{x}_{i,k} > \tau_{i,m,k}$ (P95) por 3 timesteps | Excessos univariados antecipados |
| R2 — Mahalanobis | $AS_{i,t} = \sqrt{e^\top \Sigma_i^{-1} e}$ alto | Padrões multivariados atípicos |

---

## Slide 8 — Experimentos e resultados (2 min)

> "Avaliamos o sistema em duas dimensões: qualidade da previsão e eficácia da detecção."

**Comparação de configurações (ablation study):**

| Configuração | MAE PLU | MAE Temp |
|---|---|---|
| Baseline (média vizinhos) | [valor] | [valor] |
| GraphSAGE sem FP | [valor] | [valor] |
| GraphSAGE + FP sem D⁻¹A | [valor] | [valor] |
| **GraphSAGE + FP + D⁻¹A (proposto)** | **[valor]** | **[valor]** |

> "Cada componente contribui individualmente para a melhoria."

**Evento real detectado:**
- [Descrever 1 evento real de 2024 — data, o que aconteceu, quantos timesteps antes o alerta disparou]
- Mostrar figura: série temporal com valor real, previsão e marcador de alerta

---

## Slide 9 — Conclusão (1 min)

> "Apresentamos um sistema de monitoramento climático que resolve dois problemas concretos de redes de sensores reais: lacunas nos dados e correlação espacial."

**O que foi mostrado:**
- Feature Propagation produz estimativas coerentes para nós sem dados, com custo computacional baixo
- GraphSAGE com pesos geográficos captura dependências espaço-temporais de forma eficaz
- A combinação das duas regras de detecção cobre tanto anomalias univariadas quanto multivariadas

**Limitações e trabalhos futuros:**
- Horizonte de previsão limitado pelo acúmulo de erro da GNN
- Homofilia pode ser violada em regiões com microclimas distintos
- Próximos passos: integrar dados INMET completos, explorar arquiteturas com atenção temporal

---

## Slide 10 — Referências (30 seg)

- **Hamilton et al., NeurIPS 2017** — GraphSAGE: agregação indutiva em grafos
- **Rossi et al., ICLR 2022** — Feature Propagation: interpolação via energia de Dirichlet
- **Darban et al., arXiv 2022** — Survey de detecção de anomalias em séries temporais com DL
- **Salcedo-Sanz et al., TAC 2024** — Eventos atmosféricos extremos com ML/DL
- **Mahalanobis, 1936** — Distância generalizada para anomaly score multivariado
- **Sinnott, Sky & Telescope 1984** — Fórmula de Haversine

---

## Dicas para a apresentação do artigo

- **Slide 2 (motivação):** abra com um exemplo concreto — uma data, um evento real em SP. Números e impacto humano prendem o público.
- **Slide 3 (contribuições):** leia esses três bullets em voz alta e devagar. O revisor precisa entender o que é novo antes de ver o método.
- **Slide 6 (FP):** não entre em detalhes do algoritmo — uma frase basta. O que importa é "por que FP e não média simples": a resposta é a energia de Dirichlet e a convergência para interpolação harmônica.
- **Slide 8 (resultados):** este é o slide mais importante da apresentação. Treine para falar os números de memória — não leia da tela.
- **Perguntas frequentes de revisores:**
  - *"Por que GraphSAGE e não GCN ou GAT?"* — Resposta: GraphSAGE é indutivo (generaliza para nós novos), mais adequado para redes de sensores que podem crescer.
  - *"Como você validou as anomalias detectadas?"* — Resposta: comparação com eventos registrados pela Defesa Civil / CGE em 2024.
  - *"A hipótese de homofilia é válida para São Paulo?"* — Reconheça a limitação e mencione que ilhas de calor e microclimas são uma restrição conhecida.
