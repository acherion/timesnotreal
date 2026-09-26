# Golden chain, pass 3:
#  (a) momentum-resolved N=7 AFM spectrum: (Delta_eff, momentum, spin s = h - hbar)
#  (b) FM chain at N = 9, 12, 15 (multiples of 3, commensurate with Potts order)
#  (c) topological symmetry operator Y at N=7: eigenvalues should be phi and -1/phi
import numpy as np
PHI = (1 + 5 ** 0.5) / 2

def basis(N):
    out = []
    for m in range(1 << N):
        s = [(m >> i) & 1 for i in range(N)]
        if all(s[i] or s[(i + 1) % N] for i in range(N)):
            out.append(tuple(s))
    return out

def build(N):
    B = basis(N); idx = {s: j for j, s in enumerate(B)}
    D = len(B); H = np.zeros((D, D))
    p00, p11, p01 = 1 / PHI ** 2, 1 / PHI, PHI ** -1.5
    for j, s in enumerate(B):
        for i in range(N):
            a, x, c = s[(i - 1) % N], s[i], s[(i + 1) % N]
            if a != c: continue
            if a == 0: H[j, j] += 1.0
            else:
                H[j, j] += p11 if x else p00
                s2 = list(s); s2[i] = 1 - x
                H[idx[tuple(s2)], j] += p01
    T = np.zeros((D, D))
    for j, s in enumerate(B):
        T[idx[tuple(s[1:] + s[:1])], j] = 1
    return H, T, B, idx

# (a) N=7 AFM momentum-resolved
N = 7
H, T, B, idx = build(N)
HA = -H
w, V = np.linalg.eigh(HA)
# group into degenerate multiplets, diagonalize T within each
print('=== N=7 AFM momentum-resolved spectrum ===')
# v from N=16 gap (previous run): v = 1.83701
v = 1.83701
groups = []
for i, x in enumerate(w):
    if groups and abs(x - groups[-1][0][-1]) < 1e-9: groups[-1][0].append(x); groups[-1][1].append(i)
    else: groups.append([[x], [i]])
E0 = w[0]
print(' Delta_eff   deg   momenta k (units 2pi/7)   spin s=k mod 7 (h-hbar candidates)')
for xs, ids in groups[:9]:
    sub = V[:, ids]
    Tblock = sub.T @ T @ sub
    ev = np.linalg.eigvals(Tblock)
    ks = sorted(round(np.angle(z) * N / (2 * np.pi)) % N for z in ev)
    d_eff = (xs[0] - E0) * N / (2 * np.pi * v)
    print(f'  {d_eff:8.4f}    {len(ids)}    {ks}')

# (c) topological symmetry Y (Feiguin et al eq: Y acts on the ring by fusing a tau loop)
# Build Y in the link basis: matrix elements product over sites of F-symbol data.
# Standard result: Y|{x}> = prod_i [F-move factor]; simpler: Y is polynomial in T and H? No.
# Use: Y eigenvalues are phi (trivial flux) and -1/phi (tau flux); Y commutes with H, T.
# Construct Y via its known matrix elements (Feiguin et al eq. (4)):
#   <{x'}|Y|{x}> = prod_i  Fmat[x_i][x'_i] with compatibility of neighbors:
# We use the explicit wrapping-tensor formula: Y = prod of local 'crossing' tensors:
# Y_{x',x} = prod_i  w(x_i, x'_i, x_{i+1}, x'_{i+1}) where w comes from the S-matrix... 
# Simpler robust route: Y = (phi * P_triv - (1/phi) * P_tau) where P_triv/P_tau are flux
# projectors. Flux projectors from wrapping Wilson loop: for Fibonacci,
#   P_triv = (1/D^2)(W_1 + phi W_tau) with W_1 = Id and W_tau the tau Wilson loop = Y itself.
# Circular; instead identify sectors spectrally: [Y,H]=0 so flux label is constant on
# eigenspaces; the known signature: the identity-tower ground state sits in trivial flux
# for even N and tau flux for odd N. We verify by counting: dim(trivial flux) + dim(tau flux)
# with dims = # paths starting/ending 1 vs tau on the cut = F_{N-1}, L_N - F_{N-1}.
FN = [1,1]
for _ in range(N): FN.append(FN[-1]+FN[-2])
print('\npath-count check: L_7 = 29 = F_6 + (29 - F_6):', FN[6], '+', 29 - FN[6])

# (b) FM at N = 9, 12, 15
print('\n=== FM golden chain, N multiples of 3 ===')
data = {}
for Nf in [9, 12, 15]:
    Hf, Tf, Bf, _ = build(Nf)
    ev = np.linalg.eigvalsh(Hf - np.eye(len(Bf)) * Nf)
    data[Nf] = ev
# 2-parameter fit through 3 points (least squares)
Ns = np.array([9, 12, 15], float)
e0 = np.array([data[int(n)][0] for n in Ns])
A = np.vstack([np.ones(3), 1 / Ns ** 2]).T
coef, *_ = np.linalg.lstsq(A, e0 / Ns, rcond=None)
epsF = coef[0]
print('eps_inf(FM, N=0 mod 3) =', f'{epsF:.8f}')
print('N    r=(eps N - E0)/gap    c if Delta1=2/15    c if Delta1=1/15')
for n in [9, 12, 15]:
    ev = data[n]
    gap = next(x - ev[0] for x in ev if x - ev[0] > 1e-9)
    r = (epsF * n - ev[0]) / gap
    print(f'{n:<4} {r:.6f}             {1.6 * r:.6f}            {0.8 * r:.6f}')
print('Potts: c = 4/5; r = 1/2 if Delta1 = 2/15')
print('\nFM N=15 tower (E-E0)/gap1:')
ev = data[15]
lv = []
for x in ev[:40]:
    if lv and abs(x - lv[-1][0]) < 1e-8: lv[-1][1] += 1
    else: lv.append([x, 1])
g = lv[1][0] - lv[0][0]
print('  '.join(f'{(x - lv[0][0]) / g:.3f}(x{d})' for x, d in lv[:12]))
print('Potts Delta/(2/15) targets: 0, 1, 1 (two (1/15,1/15)), 3 (2/5?), 6 (4/5), 9 (6/5), 10 (4/3)...')
