# Golden chain, pass 4: large-N sparse Lanczos convergence + N=7 phi-structure audit
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, identity
from scipy.sparse.linalg import eigsh
PHI = (1 + 5 ** 0.5) / 2

def basis(N):
    out = []
    for m in range(1 << N):
        s = [(m >> i) & 1 for i in range(N)]
        if all(s[i] or s[(i + 1) % N] for i in range(N)):
            out.append(tuple(s))
    return out

def buildH(N):
    B = basis(N); idx = {s: j for j, s in enumerate(B)}
    D = len(B); H = lil_matrix((D, D))
    p00, p11, p01 = 1 / PHI ** 2, 1 / PHI, PHI ** -1.5
    for j, s in enumerate(B):
        diag = 0.0
        for i in range(N):
            a, x, c = s[(i - 1) % N], s[i], s[(i + 1) % N]
            if a != c: continue
            if a == 0: diag += 1.0
            else:
                diag += p11 if x else p00
                s2 = list(s); s2[i] = 1 - x
                H[idx[tuple(s2)], j] += p01
        H[j, j] += diag
    return csr_matrix(H), D

# AFM large-N
print('=== AFM sparse, N up to 24 ===')
res = {}
for N in [12, 14, 16, 18, 20, 22, 24]:
    H, D = buildH(N)
    w = eigsh(-H, k=3, which='SA', return_eigenvectors=False, maxiter=5000)
    w = np.sort(w)
    res[N] = (w[0], w[1] - w[0], D)
    print(f'N={N:<3} dim={D:<7} E0={w[0]:.8f}  gap={w[1]-w[0]:.8f}')

# Richardson-style: fit eps with 1/N^2 AND 1/N^4 term over largest N
Ns = np.array(sorted(res), float)
e0 = np.array([res[int(n)][0] for n in Ns])
A = np.vstack([np.ones_like(Ns), 1 / Ns ** 2, 1 / Ns ** 4]).T
coef, *_ = np.linalg.lstsq(A, e0 / Ns, rcond=None)
eps = coef[0]
print('eps_inf (3-term fit) =', f'{eps:.9f}')
# known exact value for the AFM golden chain / A4 RSOS? print for comparison; ratio test:
print('N    r = (eps N - E0)/gap    c = 0.9 r     (target 7/9 = 0.77778, c = 0.7)')
for n in Ns:
    E0, gap, D = res[int(n)]
    r = (eps * n - E0) / gap
    print(f'{int(n):<4} {r:.6f}            {0.9 * r:.6f}')

# sequence extrapolation of c(N) (Richardson in 1/N):
cs = []
for n in Ns:
    E0, gap, D = res[int(n)]
    cs.append(0.9 * (eps * n - E0) / gap)
cs = np.array(cs)
# fit c(N) = c + a/N^b ~ assume 1/N: linear in 1/N over last 4 points
x = 1 / Ns[-4:]
p = np.polyfit(x, cs[-4:], 1)
print('c extrapolated (linear in 1/N, last 4 N):', f'{p[1]:.5f}')

# FM large-N (multiples of 3)
print('\n=== FM sparse, N = 9..24 (mult of 3) ===')
resF = {}
for N in [9, 12, 15, 18, 21, 24]:
    H, D = buildH(N)
    HF = H - identity(D) * N
    w = eigsh(HF, k=3, which='SA', return_eigenvectors=False, maxiter=5000)
    w = np.sort(w)
    resF[N] = (w[0], w[1] - w[0])
    print(f'N={N:<3} E0={w[0]:.8f}  gap={w[1]-w[0]:.8f}')
NsF = np.array(sorted(resF), float)
e0F = np.array([resF[int(n)][0] for n in NsF])
AF = np.vstack([np.ones_like(NsF), 1 / NsF ** 2, 1 / NsF ** 4]).T
coefF, *_ = np.linalg.lstsq(AF, e0F / NsF, rcond=None)
epsF = coefF[0]
print('eps_inf(FM) =', f'{epsF:.9f}')
csF = []
for n in NsF:
    E0, gap = resF[int(n)]
    r = (epsF * n - E0) / gap
    csF.append(1.6 * r)
    print(f'N={int(n):<3} r = {r:.6f}   c = 1.6 r = {1.6 * r:.6f}   (target 0.8)')
xF = 1 / NsF[-4:]
pF = np.polyfit(xF, np.array(csF)[-4:], 1)
print('c(FM) extrapolated:', f'{pF[1]:.5f}')

# N=7 phi audit: check dimensionless numbers in the exact 7-cell spectrum vs golden values
print('\n=== N=7 cell phi-structure audit ===')
H7, D7 = buildH(7)
w7 = np.linalg.eigvalsh(-H7.toarray())
E0 = w7[0]
print('E0(7) =', f'{E0:.9f}')
cands = {'-7/phi^-? none': None}
# interesting comparisons
vals = {
 '|E0|/7': abs(E0)/7, 'gap': w7[2-1]-E0 if False else None,
}
gap = next(x - E0 for x in w7 if x - E0 > 1e-9)
tests = [
 ('|E0|', abs(E0)), ('|E0|/7', abs(E0)/7), ('gap', gap), ('|E0|/gap', abs(E0)/gap),
 ('span=Emax-E0', w7[-1]-E0), ('Emax', w7[-1]),
]
golden = {'phi': PHI, 'phi^2': PHI**2, '1/phi': 1/PHI, '1/phi^2': PHI**-2, 'phi^3': PHI**3,
          'phi^4': PHI**4, '7/phi^2': 7/PHI**2, '7/phi': 7/PHI, '2phi': 2*PHI, 'sqrt5': 5**.5,
          '2phi^2': 2*PHI**2, 'phi^2/2': PHI**2/2, '4/phi': 4/PHI, 'phi^4/... none': None}
for name, val in tests:
    best = min(((k, v) for k, v in golden.items() if v), key=lambda kv: abs(kv[1]-val))
    rel = abs(best[1]-val)/val
    flag = ' <— MATCH' if rel < 2e-3 else ''
    print(f'{name:<12} = {val:.6f}   nearest {best[0]} = {best[1]:.6f}  rel dev {rel:.2%}{flag}')
