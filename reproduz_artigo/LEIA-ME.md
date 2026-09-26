# Scripts que reproduzem os números e as figuras do artigo

Rodar **a partir da raiz do repositório**:

```bash
python reproduz_artigo/<script>.py
```

Todos leem `artifacts/X_test.pkl` + `artifacts/graph_data.pkl`, gerados pelo
notebook. Os caminhos vêm de `caminhos.py`, que resolve a raiz a partir da
posição do próprio script: mover esta pasta de lugar quebra tudo, mover as
outras pastas não.

> ### ⚠ Erro de leitura de datas nos resultados publicados (corrigido nesta branch)
>
> A leitura dos CSVs usava `pd.to_datetime(..., format="mixed", dayfirst=True)`.
> Os CSVs do CGE usam datas ISO (`AAAA-MM-DD`), e no **pandas 3** (versão do
> `requirements.txt`) essa combinação **troca dia e mês sempre que o dia é
> ≤ 12**: `2024-01-02` vira 1º de fevereiro. No pandas 2 a leitura sai certa.
> Os resultados publicados (Latin.Science 2026 e relatório IT) foram gerados com
> o pandas 3, então **36% das linhas de 2016–2024** (dias 1 a 12 de cada mês)
> entraram com a data trocada, no treino e no teste.
>
> Confirmado reexecutando as células 0–15 do notebook: com pandas 3.0.3 e a
> leitura antiga saem exatamente os 97 dias de evento (107 datando por `t`) e
> as médias do nó cego 35 salvas no notebook; com as datas corretas são 80
> dias. Exemplos do efeito:
> - 42 dos 97 dias de evento só existem por causa da troca: são saltos
>   artificiais do contador de chuva na emenda entre dias trocados (00h–03h);
> - o "pico de 61,5 mm" de 13/01 na Barragem Palheiros comparava a leitura de
>   00h50 com a de 23h50 de **1º/12** (0,25 mm). O incremento real às 00h50 é
>   30,0 mm, e a hora mais chuvosa foi 00h–01h, com 43,5 mm.
>
> A correção é `format="ISO8601"` na célula 7 do notebook, em `missing.py` e em
> `regen_x_test.py`. Como o treino também usou datas trocadas, **todos os
> artefatos precisam ser regerados**, o que exige retreinar. O
> `artifacts/X_test.pkl` versionado hoje foi regenerado em agosto (com pandas 2,
> datas corretas, mas modelos antigos) e também deve ser substituído.
>
> **Até rodar o roteiro abaixo, não rode `fig_mapa.py` nem `fig_evento.py`**:
> elas sobrescreveriam as figuras do relatório com dados inconsistentes.

## Roteiro para regerar todos os resultados (máquina com GPU)

Na raiz do repositório, com os CSVs de 2016–2024 em
`estacoes_meteorologicas/meteorologicas_<ano>/` (2016–2022 não são versionados):

```bash
pip install -r requirements.txt

# 1. Confere a leitura das datas. Tem que terminar em "OK"; se falhar, pare.
python reproduz_artigo/verifica_datas.py

# 2. Notebook completo (Restart & Run All): regera artifacts/X_test.pkl,
#    model_plu.pt, model_temp.pt, tau.pkl, sigma_inv.pkl e graph_data.pkl.
jupyter nbconvert --to notebook --execute --inplace grafo-0.8.0-pyg.ipynb

# 3. Caches por ano, usados pela ablação e pelo STGCN.
python reproduz_artigo/prep_peryear.py

# 4. Ablação e STGCN (3 sementes cada; é a etapa mais demorada).
python reproduz_artigo/ablacao_valsplit.py
python reproduz_artigo/stgcn_valsplit.py

# 5. Números do artigo (cada script imprime os valores das tabelas).
mkdir -p resultados_rerun
for s in numeros regime por_estacao tabela_iag fp_mascaramento fp_mascaramento_krig \
         deteccao_reforcada deteccao_por_estacao regra1_antecedencia; do
    python reproduz_artigo/$s.py > "resultados_rerun/$s.txt"
done

# 6. Figuras (só depois dos passos acima).
python reproduz_artigo/fig_evento.py
python reproduz_artigo/fig_mapa.py
```

Guarde a pasta `resultados_rerun/` e os `.json` gerados em `artifacts/`: são
eles que alimentam a atualização do texto do artigo.

| script | o que faz |
|---|---|
| `numeros.py`   | Tabelas II e IV + correlação + estudo de caso 13/01. **Convenção: erro AGREGADO sobre os 32 nós observados** (a persistência bate exata assim; por-nó NÃO bate). |
| `tabela_iag.py` | Tabela III: nó cego 33 vs medições reais do IAG-USP (diário, 2024). Lê as planilhas do IAG se disponíveis; senão usa o CSV derivado `dados_iag_2024_diario.csv` (versionado). |
| `regime.py`    | Estratificação do incremento de chuva por regime (seco/moderado/forte): GraphSAGE vs persistência. |
| `por_estacao.py` | MAE/RMSE por estação (chuva e temperatura) + mín/mediana/máx do RMSE. |
| `fig_evento.py`| `figs/evento_20240113.{pdf,png}` |
| `fig_mapa.py`  | `figs/mapa_alertas_20240113.{pdf,png}` — usa `distritos.geojson` (96 distritos de SP, CRS84). Não precisa geopandas. |
| `missing.py`   | % de dados ausentes por ano na grade de 10 min. Uso: `missing.py 2016 ... 2024` |
| `pesos.py`     | Prova que 1/d == D^-1A na GNN e que o FP precisa da soma = 1 |
| `didatico.py`  | Exemplo mínimo de 3 vizinhos explicando a normalização |
| `stgcn_valsplit.py` | STGCN (janela W=12 = 2h, 2 blocos T-conv GLU + conv espacial D^-1A) sob o MESMO protocolo da ablação valsplit (2016-22/23/24, 3 sementes × PLU/Temp) → `artifacts/stgcn_valsplit.json`. Lê `peryear_cache.pt`. Único modelo que bate a persistência no RMSE de chuva (2,818 vs 2,856); Temp 0,521/0,929. `--smoke` p/ teste rápido. |

## Convenção crítica: as séries são indexadas por t+6, não por t

`trues_chuva_all[i]` é o acumulado em `t+6`. O incremento de 60 min é
`trues_chuva_all - X_2024[:T,:,0]`, e deve ser datado pelo instante **ALVO**
(`timestamps_2024[6:T+6]`), não pelo de emissão. Consequências:
- o incremento que chega em t+6 é a chuva da hora terminada em t+6, e é assim
  que se contam os dias de evento ("algum posto > 10mm/60min no dia");
- por isso o eixo x da figura do evento começa em **23h** (a chuva de 13/01 00h é prevista às 12/01 23h);
- os valores antes citados aqui (97/366 dias; pico de 61,5 mm em 13/01) vinham
  das datas trocadas (ver o aviso acima). Com as datas corretas: 80/366 dias e
  30,0 mm às 00h50 de 13/01 (hora mais chuvosa: 00h–01h, 43,5 mm).

Fonte do geojson: https://github.com/codigourbano/distritos-sp
