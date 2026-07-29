# Exemplo minimo: um no X com 3 vizinhos.
#   A a  1 km, marcando 20 C
#   B a  2 km, marcando 22 C
#   C a 10 km, marcando 30 C
# A resposta "certa" tem que ficar entre 20 e 30, puxada para o A (o mais perto).

dist = {"A": 1.0, "B": 2.0, "C": 10.0}
temp = {"A": 20.0, "B": 22.0, "C": 30.0}

# ---------- os tres jeitos de definir o peso ----------
bruto = {k: 1 / d for k, d in dist.items()}                    # 1/d, sem normalizar

S = sum(bruto.values())
d_inv_a = {k: w / S for k, w in bruto.items()}                 # D^-1 A  = divide pela soma

GMAX, GMIN = 10.0, 0.035                                       # max/min globais do grafo real
minmax = {k: (w - GMIN) / (GMAX - GMIN) for k, w in bruto.items()}  # subtrai o minimo

metodos = [("1/d bruto", bruto), ("D^-1A", d_inv_a), ("min-max", minmax)]

print("=" * 66)
print("PASSO 1 - quanto vale o peso de cada vizinho em cada metodo")
print("=" * 66)
print(f"{'':12} {'A (1km)':>10} {'B (2km)':>10} {'C (10km)':>10} {'SOMA':>10}")
for nome, w in metodos:
    print(f"{nome:12} {w['A']:10.4f} {w['B']:10.4f} {w['C']:10.4f} {sum(w.values()):10.4f}")

print()
print("PASSO 2 - as PROPORCOES entre os pesos (dividi tudo pelo peso do C)")
print("          se as proporcoes mudam, o metodo distorceu a vizinhanca")
for nome, w in metodos:
    r = {k: v / w["C"] for k, v in w.items()}
    print(f"  {nome:12} A:B:C = {r['A']:5.1f} : {r['B']:5.1f} : {r['C']:4.1f}")

print()
print("=" * 66)
print("PASSO 3 - a GNN: ela DIVIDE pela soma dos pesos (media ponderada)")
print("          mensagem = soma(w * temp) / soma(w)")
print("=" * 66)
for nome, w in metodos:
    num = sum(w[k] * temp[k] for k in dist)
    den = sum(w.values())
    print(f"  {nome:12}: {num:8.4f} / {den:6.4f} = {num/den:7.4f} C")

print()
print("  -> 1/d e D^-1A dao o MESMO valor. Dividir todos os pesos por 1.6")
print("     divide o de cima por 1.6 E o de baixo por 1.6. Cancela.")
print("  -> min-max da outro valor, porque SUBTRAIR o minimo muda as proporcoes.")

print()
print("=" * 66)
print("PASSO 4 - o Feature Propagation: ele NAO divide, so soma")
print("          x_novo = soma(w * temp)")
print("=" * 66)
for nome, w in metodos:
    v = sum(w[k] * temp[k] for k in dist)
    ok = "OK, entre 20 e 30" if 20 <= v <= 30 else "<-- FORA DA FAIXA FISICA!"
    print(f"  {nome:12}: {v:8.4f} C   {ok}")

print()
print("  -> Aqui a soma dos pesos PRECISA dar 1, senao o valor infla ou encolhe.")
print("     So o D^-1A garante isso.")
