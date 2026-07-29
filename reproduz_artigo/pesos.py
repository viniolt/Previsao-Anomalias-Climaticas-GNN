"""Demonstra as afirmacoes da Secao III-A do artigo sobre normalizacao de pesos:
(1) na agregacao do GraphSAGE, 1/d bruto e D^-1A produzem saidas identicas;
(2) no Feature Propagation (que NAO divide pela soma), so o D^-1A garante
    combinacao convexa (soma incidente = 1); a simetrica encolhe (0,792 no
    no 33) e o min-max infla (1,409).

Le o grafo canonico dos experimentos em artifacts/graph_data.pkl (003.graphml),
recomputando os pesos brutos 1/d por Haversine sobre as coordenadas reais.
"""
import os, pickle, math
import torch

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
g = pickle.load(open(os.path.join(ART, "graph_data.pkl"), "rb"))
nodes = g["nodes_lista"]; coords = g["coords"]; labels = g["labels"]
N = len(nodes)
edge_index = torch.as_tensor(g["edge_index"])


def hav(a, b):
    R = 6371
    la1, lo1 = map(math.radians, a); la2, lo2 = map(math.radians, b)
    x = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2 * R * math.asin(math.sqrt(x))


raw = []
for e in range(edge_index.size(1)):
    u, v = nodes[edge_index[0, e]], nodes[edge_index[1, e]]
    raw.append(1.0 / max(hav(coords[u], coords[v]), 0.1))
w_raw = torch.tensor(raw, dtype=torch.float64)
dists = [1.0 / w for w in raw]
print(f"grafo: {N} nos, {edge_index.size(1)//2} arestas nao direcionadas, "
      f"dist {min(dists):.2f}-{max(dists):.2f} km\n")


def norm_destino(ew):                       # D^-1 A  (o metodo do artigo)
    deg = torch.zeros(N, dtype=torch.float64).index_add_(0, edge_index[1], ew)
    return ew / deg.clamp(min=1e-12)[edge_index[1]]


def norm_sym(ew):                           # D^-1/2 A D^-1/2 (o bug antigo)
    deg = torch.zeros(N, dtype=torch.float64).index_add_(0, edge_index[1], ew)
    d = deg.clamp(min=1e-12).pow(-0.5)
    return d[edge_index[0]] * ew * d[edge_index[1]]


def minmax(ew):                             # min-max global (mais antigo ainda)
    return (ew - ew.min()) / (ew.max() - ew.min())


def conv(ew, x):
    """Reproduz exatamente a WeightedSAGEConv: (sum_j w_ij x_j) / (sum_j w_ij)."""
    deg = torch.zeros(N, dtype=torch.float64).index_add_(0, edge_index[1], ew)
    deg = deg.clamp(min=1e-8)
    agg = torch.zeros(N, x.size(1), dtype=torch.float64)
    agg.index_add_(0, edge_index[1], ew.view(-1, 1) * x[edge_index[0]])
    return agg / deg.unsqueeze(-1)


torch.manual_seed(0)
x = torch.randn(N, 4, dtype=torch.float64)          # features ficticias

o_raw = conv(w_raw, x)
o_dst = conv(norm_destino(w_raw), x)
o_mm = conv(minmax(w_raw), x)

print("=== Saida da agregacao do GraphSAGE (a mensagem dos vizinhos) ===")
print(f"  1/d  vs  D^-1A : diferenca maxima = {(o_raw-o_dst).abs().max():.3e}  <-- ZERO")
print(f"  1/d  vs  min-max: diferenca maxima = {(o_raw-o_mm ).abs().max():.3e}  <-- MUDA")
print()

print("=== Por que: soma dos pesos que chegam em cada no ===")
for nome, ew in [("1/d bruto", w_raw), ("D^-1A", norm_destino(w_raw)), ("min-max", minmax(w_raw))]:
    deg = torch.zeros(N, dtype=torch.float64).index_add_(0, edge_index[1], ew)
    print(f"  {nome:10s}: soma por no varia de {deg.min():.4f} a {deg.max():.4f}")
print()

no33 = nodes.index("33")
print("=== E no Feature Propagation (que NAO divide pela soma)? ===")
print("Soma dos pesos que chegam no no cego 33 (deve ser 1.0 p/ combinacao convexa):")
for nome, ew in [("1/d bruto", w_raw), ("D^-1A (artigo)", norm_destino(w_raw)),
                 ("simetrica (bug)", norm_sym(w_raw)), ("min-max (antigo)", minmax(w_raw))]:
    s = sum(float(ew[e]) for e in range(edge_index.size(1)) if edge_index[1, e] == no33)
    flag = "  <-- media ponderada OK" if abs(s - 1) < 1e-9 else f"  <-- X sai da faixa fisica ({s:.2f}x)"
    print(f"  {nome:18s}: soma = {s:6.3f}{flag}")
