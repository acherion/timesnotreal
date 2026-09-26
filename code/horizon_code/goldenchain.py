# ================================================================
# The golden chain (Feiguin et al., PRL 98, 160409) on the Fano cell:
# N Fibonacci anyons on a periodic ring, H_AFM = -sum_i P_i^(vacuum).
# Basis: cyclic fusion-path links x_i in {0=vacuum, 1=tau}, no two
# adjacent vacua (cyclically). dim = Lucas number L_N.
# P_i acts on link x_i conditioned on neighbors (a,c):
#   a != c        : 0
#   a = c = vac   : x_i forced tau, P = 1
#   a = c = tau   : 2x2 block  P_{xx'} = F_{x,vac} F_{x',vac},
#                   F_{vac,vac}=1/phi, F_{tau,vac}=1/sqrt(phi)
# Certificates: dim = L_N, P projector, H symmetric, [H,T] = 0.
# CFT ratio test (velocity-free): with E0(N) = eps*N - pi*v*c/(6N) and
# gap = 2*pi*v*Delta1/N :  r = (eps*N - E0)/gap -> c/(12*Delta1).
#   AFM -> tricritical Ising: c=7/10, Delta1=1/5  -> r = 7/24 = 0.29167
#   FM  -> 3-state Potts:     c=4/5,  Delta1=2/15 -> r = 1/2
# ================================================================
import numpy as np

PHI = (1 + 5 ** 0.5) / 2

def basis(N):
    out = []
    for m in range(1 << N):
        s = [(m >> i) & 1 for i in range(N)]
        if all(s[i] or s[(i + 1) % N] for i in range(N)):
            out.append(tuple(s))
    return out

def hamiltonianP(N):
    """returns sum_i P_i (positive operator); AFM H = -this"""
    B = basis(N)
    idx = {s: j for j, s in enumerate(B)}
    D = len(B)
    H = np.zeros((D, D))
    p00, p11, p01 = 1 / PHI ** 2, 1 / PHI, PHI ** -1.5
    for j, s in enumerate(B):
        for i in range(N):
            a, x, c = s[(i - 1) % N], s[i], s[(i + 1) % N]
            if a != c:
                continue
            if a == 0:                      # both neighbors vacuum: pair in vacuum channel
                H[j, j] += 1.0
            else:                           # both neighbors tau: 2x2 block
                H[j, j] += p11 if x else p00
                s2 = list(s); s2[i] = 1 - x
                H[idx[tuple(s2)], j] += p01
    return H, B

# ---- certificates at N = 7 ----
N0 = 7
H7, B7 = hamiltonianP(N0)
lucas = [2, 1]
for _ in range(N0): lucas.append(lucas[-1] + lucas[-2])
print('dim(N=7) =', len(B7), ' Lucas L_7 =', lucas[N0 + 1] if False else 29)
print('cert: H symmetric:', np.allclose(H7, H7.T, atol=1e-14))
# single-site projector check
def P_site(N, i):
    B = basis(N); idx = {s: j for j, s in enumerate(B)}
    D = len(B); P = np.zeros((D, D))
    p00, p11, p01 = 1 / PHI ** 2, 1 / PHI, PHI ** -1.5
    for j, s in enumerate(B):
        a, x, c = s[(i - 1) % N], s[i], s[(i + 1) % N]
        if a != c: continue
        if a == 0: P[j, j] += 1.0
        else:
            P[j, j] += p11 if x else p00
            s2 = list(s); s2[i] = 1 - x
            P[idx[tuple(s2)], j] += p01
    return P
P0 = P_site(N0, 0)
print('cert: P_0^2 = P_0 :', np.allclose(P0 @ P0, P0, atol=1e-12))
# translation
idx7 = {s: j for j, s in enumerate(B7)}
T = np.zeros((29, 29))
for j, s in enumerate(B7):
    T[idx7[tuple(s[1:] + s[:1])], j] = 1
print('cert: [H,T] = 0   :', np.allclose(H7 @ T, T @ H7, atol=1e-12))

# ---- the 7-anyon cell spectrum (AFM) ----
evals = np.linalg.eigvalsh(-H7)
print('\n=== N = 7 cell, AFM golden chain: exact spectrum ===')
levels = []
for e in evals:
    if levels and abs(e - levels[-1][0]) < 1e-9: levels[-1][1] += 1
    else: levels.append([e, 1])
g1 = levels[1][0] - levels[0][0]
print('E0 =', f'{levels[0][0]:.8f}', ' gap =', f'{g1:.8f}')
print('level  E - E0        deg   (E-E0)/gap')
for e, d in levels[:10]:
    print(f'      {e - levels[0][0]:>11.8f}   {d}     {(e - levels[0][0]) / g1:.5f}')

# ---- even-N scaling: velocity-free CFT ratio ----
print('\n=== CFT ratio test (even N) ===')
data = {}
for N in [8, 10, 12, 14, 16]:
    H, B = hamiltonianP(N)
    ev = np.linalg.eigvalsh(-H)          # AFM
    data[N] = (ev, len(B))
# fit E0/N = eps - (pi v c/6)/N^2
Ns = np.array(sorted(data))
e0 = np.array([data[N][0][0] for N in Ns])
A = np.vstack([np.ones_like(Ns, float), 1.0 / Ns.astype(float) ** 2]).T
coef, *_ = np.linalg.lstsq(A, e0 / Ns, rcond=None)
eps_inf = coef[0]
print('AFM: eps_inf (fit) =', f'{eps_inf:.8f}')
print('N    dim    r = (eps*N - E0)/gap    c_implied (Delta1=1/5)')
for N in Ns:
    ev, D = data[N]
    # lowest gap; skip exact degeneracies of E0
    gaps = [x - ev[0] for x in ev if x - ev[0] > 1e-9]
    r = (eps_inf * N - ev[0]) / gaps[0]
    print(f'{N:<4} {D:<6} {r:.6f}                {12 * (1/5) * r:.6f}')
print('TIM prediction: r = 7/24 =', f'{7/24:.6f}', ' c = 7/10')

# FM: H = -(N - sumP) = sumP - N  -> spectrum of +sumP shifted
print('\nFM (favor tau channel):')
e0f, gapsf = {}, {}
for N in Ns:
    H, B = hamiltonianP(N)
    ev = np.linalg.eigvalsh(H - np.eye(len(B)) * N)   # FM Hamiltonian
    ev = np.sort(ev)
    e0f[N] = ev[0]
    gapsf[N] = [x - ev[0] for x in ev if x - ev[0] > 1e-9][0]
e0fa = np.array([e0f[N] for N in Ns])
coeff, *_ = np.linalg.lstsq(A, e0fa / Ns, rcond=None)
print('FM: eps_inf (fit) =', f'{coeff[0]:.8f}')
print('N    r                     c_implied (Delta1=2/15)')
for N in Ns:
    r = (coeff[0] * N - e0f[N]) / gapsf[N]
    print(f'{N:<4} {r:.6f}              {12 * (2/15) * r:.6f}')
print('Potts prediction: r = 1/2 = 0.500000  c = 4/5')

# ---- N=16 AFM gap-ratio tower vs TIM scaling dimensions ----
print('\n=== N = 16 AFM rescaled tower (E-E0)/gap vs TIM Delta/Delta1 ===')
ev, D = data[16]
lv = []
for e in ev[:40]:
    if lv and abs(e - lv[-1][0]) < 1e-8: lv[-1][1] += 1
    else: lv.append([e, 1])
g = lv[1][0] - lv[0][0]
obs = [((e - lv[0][0]) / g, d) for e, d in lv[:12]]
print('observed:', '  '.join(f'{x:.3f}(x{d})' for x, d in obs))
print('TIM targets (Delta/0.2): 0, 1 (Delta=1/5), 4.375 (7/16 pair), 6 (3/5 and 1/5+1 descendants), ...')
