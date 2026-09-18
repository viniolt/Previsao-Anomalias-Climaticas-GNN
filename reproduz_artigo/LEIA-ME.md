# Scripts que reproduzem os números e as figuras do artigo

Rodar **a partir da raiz do repositório**:

```bash
python reproduz_artigo/<script>.py
```

Todos leem `artifacts/X_test.pkl` + `artifacts/graph_data.pkl`, gerados pelo
notebook. Os caminhos vêm de `caminhos.py`, que resolve a raiz a partir da
posição do próprio script: mover esta pasta de lugar quebra tudo, mover as
outras pastas não.

> ### ⚠ O `artifacts/X_test.pkl` atual NÃO é o que gerou os números publicados
>
> O pickle no repositório foi **regenerado** por `regen_x_test.py` (agosto de
> 2026) porque o original não estava versionado. Ele bate com o artigo em
> várias coisas (registros brutos de 2024, frações de dados ausentes, FP do nó
> 33, correlação de temperatura), mas **diverge** em outras: 80 dias de evento
> em vez de 97, persistência PLU 0,286 em vez de 0,359, precisão p99 21,9% em
> vez de 26,6%. A causa provável é dependência de versão do pandas no
> `interpolate`/`resample` do pré-processamento.
>
> Consequências práticas:
>
> - **Não rode `fig_mapa.py` nem `fig_evento.py`.** As figuras em
>   `relatorio-it/figs/` são as originais (14/08) e sobrescrevê-las produziria
>   imagens que contradizem o texto do relatório.
> - `fig_grafo.py` é seguro: depende só de `graph_data.pkl`, que é versionado.
> - Os números impressos por `numeros.py`, `regime.py`, `por_estacao.py` e
>   `deteccao_*.py` serão os regenerados, não os publicados.
> - Recuperar o original exige rodar o notebook inteiro na máquina de origem,
>   com os CSVs de 2016–2022, que não estão neste repositório.

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
- evento = "algum posto > 10mm/60min no dia" dá **97/366** por t+6 (por t dá 107, errado);
- o pico de 13/01 em Barragem Palheiros dá **61,5 mm** por t+6 (por t dá 43,5, errado);
- por isso o eixo x da figura do evento começa em **23h** (a chuva de 13/01 00h é prevista às 12/01 23h).

Fonte do geojson: https://github.com/codigourbano/distritos-sp
