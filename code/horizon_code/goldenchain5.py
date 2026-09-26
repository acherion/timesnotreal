# Pass 5: is eps_inf(AFM) exactly -2/phi^2 ?
#  (a) independent exact 3-point solves (no least squares) at different N windows
#  (b) push to N=26,28 for a final window
#  (c) Bethe-ansatz cross-check: TL chain <-> XXZ at Delta = -cos(pi/5);
#      exact XXZ energy density integral -> map back to TL normalization
#  (d) structure search on FM value
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, identity
from scipy.sparse.linalg import eigsh
from scipy.integrate import quad
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

E0 = {}
for N in [16, 18, 20, 22, 24, 26]:
    H, D = buildH(N)
    w = eigsh(-H, k=1, which='SA', return_eigenvectors=False, maxiter=8000, tol=0)
    E0[N] = w[0]
    print(f'N={N:<3} dim={D:<7} E0={w[0]:.12f}')

def solve3(n1, n2, n3):
    # e0/N = eps + b/N^2 + c/N^4 exact through 3 points
    A = np.array([[1, 1/n1**2, 1/n1**4], [1, 1/n2**2, 1/n2**4], [1, 1/n3**2, 1/n3**4]], float)
    y = np.array([E0[n1]/n1, E0[n2]/n2, E0[n3]/n3])
    return np.linalg.solve(A, y)[0]

target = -2 / PHI ** 2
print('\ntarget -2/phi^2 =', f'{target:.12f}')
for tri in [(16, 20, 24), (18, 22, 26), (20, 22, 24), (22, 24, 26)]:
    eps = solve3(*tri)
    print(f'window {tri}: eps = {eps:.12f}   dev from -2/phi^2 = {eps - target:.3e}')

# (c) Bethe ansatz: TL_n(d=2cos g), g = pi/5. Mapping to XXZ:
# H_TL = -sum e_i ; standard result: e_i = (1/2)[...] maps to XXZ with anisotropy
# Delta = -cos g, and E_TL = E_XXZ_shifted. Known exact gs energy of
# H_XXZ = sum_i [SxSx + SySy + Delta SzSz]  (critical, -1<Delta<1, Delta = cos mu, mu = pi - g):
#   e_xxz(mu) = Delta/4 - (sin^2 mu)/2 * I,  I = integral dx sech(pi x) / (cosh(2 mu x) - cos mu)
# TL <-> XXZ: e_i = 1/2 - (S_i.S_{i+1})... more precisely
#   e_i = -(Sx Sx + Sy Sy)_i,i+1 - cos g * (Sz Sz)_{i,i+1} + cos g /4  - (i sin g /2)(Sz_i - Sz_{i+1})?? 
# Use the cleaner known identity: spectrum of -sum e_i on the TL chain equals spectrum of
# XXZ with Delta = -cos g up to shift: H_TL = H_XXZ(Delta=-cos g) - N*(d/4) with d=2cos g? 
# Rather than trust memory, verify numerically at small N with exact diagonalization of XXZ.
def xxz_spec(N, Delta):
    dim = 1 << N
    H = np.zeros((dim, dim))
    for m in range(dim):
        for i in range(N):
            j = (i + 1) % N
            si, sj = (m >> i) & 1, (m >> j) & 1
            H[m, m] += Delta * (0.25 if si == sj else -0.25)
            if si != sj:
                m2 = m ^ (1 << i) ^ (1 << j)
                H[m2, m] += 0.5
    return np.linalg.eigvalsh(H)

g = np.pi / 5
for N in [8, 10]:
    H, D = buildH(N)
    tl = np.linalg.eigvalsh(-H.toarray())
    for Delta in [-np.cos(g), np.cos(g)]:
        for shift_name, shift in [('-N cos(g)/2... try match', None)]:
            pass
    xz = xxz_spec(N, -np.cos(g))
    # compare ground states with an additive shift guess: TL e0 - XXZ e0
    print(f'N={N}: TL_E0 = {tl[0]:.8f}  XXZ(-cos g)_E0 = {xz[0]:.8f}  diff/N = {(tl[0]-xz[0])/N:.8f}')
    # also gap comparison to see if spectra align (twisted sectors differ; just report)
    print(f'      TL gap = {tl[1]-tl[0]:.6f}  XXZ gap = {xz[1]-xz[0]:.6f}')

# exact XXZ gs energy density at Delta = -cos(pi/5): standard formula
# for Delta = cos(mu) with 0<mu<pi: e = cos(mu)/4 - sin(mu)/2 * integral_{-inf}^{inf} dx sech(pi x)* [1 - cos(mu)... ]
# canonical: e(mu) = Delta/4 - sin^2(mu) * I2, I2 = (1/2)*int dx / (cosh(pi x)*(cosh(2 mu x) - cos mu))
mu = np.pi - g   # Delta = -cos g = cos(pi - g)
Delta = np.cos(mu)
I2, err = quad(lambda x: 1.0 / (np.cosh(np.pi * x) * (np.cosh(2 * mu * x) - np.cos(mu))), -50, 50, limit=400)
e_xxz = Delta / 4 - np.sin(mu) ** 2 / 2 * I2
print('\nXXZ exact gs density at Delta = -cos(pi/5):', f'{e_xxz:.12f}')
# then TL density = e_xxz + (diff/N measured above)
diffN = None
N = 10
H, D = buildH(N)
tl0 = eigsh(-H, k=1, which='SA', return_eigenvectors=False)[0]
xz0 = xxz_spec(N, -np.cos(g))[0]
print('measured shift/N at N=10:', f'{(tl0 - xz0)/N:.9f}', ' candidates: -cos(g)/2 =', f'{-np.cos(g)/2:.9f}',
      ' -d/4 =', f'{-np.cos(g)/2:.9f}', ' -1/2 - cos(g)/4? =', f'{-0.5 - np.cos(g)/4:.9f}')
print('predicted eps_TL = e_xxz + shift/N =', f'{e_xxz + (tl0 - xz0)/N:.9f}', ' (approx; shift may have 1/N part)')
print('compare -2/phi^2 =', f'{target:.9f}')

# (d) FM structure search
valF = 0.935183947
print('\n=== FM eps structure search: value', valF, '===')
import itertools
names = {'1': 1.0, '1/phi': 1/PHI, '1/phi^2': PHI**-2, '1/phi^3': PHI**-3, '1/phi^4': PHI**-4,
         'sqrt5': 5**0.5, '1/sqrt5': 5**-0.5, 'pi': np.pi, '1/pi': 1/np.pi,
         '3sqrt3/2pi': 3*np.sqrt(3)/(2*np.pi), 'sqrt3': np.sqrt(3), 'cos(pi/5)': np.cos(np.pi/5),
         'sin(pi/5)': np.sin(np.pi/5), 'cos(pi/10)': np.cos(np.pi/10), 'sin(pi/10)': np.sin(np.pi/10),
         'sqrt(2+phi)': np.sqrt(2+PHI), '1/sqrt(2+phi)': 1/np.sqrt(2+PHI)}
hits = []
for (n1, v1), (n2, v2) in itertools.product(names.items(), repeat=2):
    for a in [-3,-2,-1,-0.5,0.5,1,2,3]:
        for b in [-3,-2,-1,-0.5,0,0.5,1,2,3]:
            t = a*v1 + b*v2
            if abs(t - valF) < 5e-7:
                hits.append(f'{a}*{n1} + {b}*{n2} = {t:.9f}')
seen = set()
for h in hits[:20]:
    if h not in seen: print('  ', h); seen.add(h)
if not hits: print('   no simple 2-term hit at 5e-7')
