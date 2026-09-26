# Emergent relativity, the attemptable fragment:
# The golden chain (Fibonacci anyons, AFM) should flow to the tricritical Ising
# CFT. A CFT is exactly Lorentz-invariant: E_n - E0 = (2 pi v / L) * x_n with
# ONE velocity v for all states (the emergent "c"), and universal exponents x_n.
# Test: diagonalize the chain, resolve momenta, and check
#   (a) gap ratio [k=0 gap]/[k=pi gap] = (1/5)/(3/40) = 8/3   (v cancels)
#   (b) one v fits all towers
#   (c) central charge from E0(L) scaling  ->  c = 7/10
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

PHI = (1 + 5**0.5) / 2

def basis(L):
    states = []
    for s in range(1 << L):
        ok = True
        for i in range(L):
            if not (s >> i) & 1 and not (s >> ((i + 1) % L)) & 1:
                ok = False; break
        if ok: states.append(s)
    return states

def build_H(L, states, idx):
    rows, cols, vals = [], [], []
    for a, s in enumerate(states):
        for i in range(L):
            l = (s >> ((i - 1) % L)) & 1
            m = (s >> i) & 1
            r = (s >> ((i + 1) % L)) & 1
            if l == 0 and r == 0:
                rows.append(a); cols.append(a); vals.append(-1.0)
            elif l == 1 and r == 1:
                diag = -PHI**-2 if m == 0 else -PHI**-1
                rows.append(a); cols.append(a); vals.append(diag)
                b = idx[s ^ (1 << i)]
                rows.append(b); cols.append(a); vals.append(-PHI**-1.5)
    return coo_matrix((vals, (rows, cols)), shape=(len(states),) * 2).tocsr()

def momenta(L, states, idx, vecs, E):
    # translation operator as index permutation
    perm = np.empty(len(states), dtype=np.int64)
    for a, s in enumerate(states):
        perm[a] = idx[((s << 1) | (s >> (L - 1))) & ((1 << L) - 1)]
    Tv = np.zeros_like(vecs)
    Tv[perm, :] = vecs
    ks = []
    used = np.zeros(len(E), bool)
    for i in range(len(E)):
        if used[i]: continue
        cluster = [j for j in range(len(E)) if abs(E[j] - E[i]) < 1e-9]
        for j in cluster: used[j] = True
        block = vecs[:, cluster].T @ Tv[:, cluster]
        for lam in np.linalg.eigvals(block):
            ks.append((E[i], math.atan2(lam.imag, lam.real)))
    return ks

print(f'{"L":>3} {"dim":>6} {"E0/L":>10} {"v(k=0)":>8} {"ratio(8/3=2.667)":>16}')
res = {}
for L in (18, 20, 22):
    st = basis(L)
    idx = {s: a for a, s in enumerate(st)}
    H = build_H(L, st, idx)
    E, V = eigsh(H, k=16, which='SA')
    order = np.argsort(E)
    E, V = E[order], V[:, order]
    ks = momenta(L, st, idx, V, E)
    E0 = E[0]
    # lowest gap at k=0 (excluding ground state) and at k=pi
    g0 = min(e - E0 for e, k in ks if abs(k) < 0.1 and e - E0 > 1e-9)
    gpi = min(e - E0 for e, k in ks if abs(abs(k) - math.pi) < 0.1)
    v = g0 * L / (2 * math.pi) / (1 / 5)        # if x(k=0) = 1/5
    res[L] = (E0 / L, v, g0 / gpi)
    print(f'{L:>3} {len(st):>6} {E0/L:>10.6f} {v:>8.4f} {g0/gpi:>16.4f}')

# central charge from E0(L) = eps*L - pi*c*v/(6L), using consecutive L pairs
(L1, L2, L3) = (18, 20, 22)
import itertools
print()
for (La, Lb) in ((L1, L2), (L2, L3)):
    ea, va = res[La][0], res[La][1]
    eb, vb = res[Lb][0], res[Lb][1]
    # E0/L = eps - (pi c v/6) / L^2  -> solve 2 eqs
    cva = (ea - eb) / (1 / La**2 - 1 / Lb**2) * (-6 / math.pi)
    vmean = (va + vb) / 2
    print(f'  L=({La},{Lb}): c*v = {cva:.4f}, v = {vmean:.4f}  ->  c = {cva/vmean:.4f}   (target 0.7)')
