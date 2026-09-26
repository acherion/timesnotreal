# Pass 6: final precision test of eps_inf = -2/phi^2 = sqrt(5) - 3
# 4-point exact solves with terms 1, 1/N^2, 1/N^4, 1/N^6 across windows,
# using tight-tolerance Lanczos E0 values (recomputed at tol=1e-14).
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
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
    return csr_matrix(H)

E0 = {}
for N in [18, 20, 22, 24, 26, 28]:
    H = buildH(N)
    w, v = eigsh(-H, k=1, which='SA', maxiter=20000, tol=1e-14)
    # Rayleigh refinement
    x = v[:, 0]
    E0[N] = float(x @ (-H @ x)) / float(x @ x)
    print(f'N={N:<3} E0 = {E0[N]:.14f}')

target = 5 ** 0.5 - 3
print('\ntarget sqrt(5) - 3 = -2/phi^2 =', f'{target:.14f}')
def solve4(ns):
    A = np.array([[1, 1/n**2, 1/n**4, 1/n**6] for n in ns], float)
    y = np.array([E0[n]/n for n in ns])
    return np.linalg.solve(A, y)[0]
for win in [(18, 20, 22, 24), (20, 22, 24, 26), (22, 24, 26, 28)]:
    eps = solve4(win)
    print(f'window {win}: eps = {eps:.14f}  dev = {eps - target:+.3e}')
# also 3-point windows on largest
def solve3(ns):
    A = np.array([[1, 1/n**2, 1/n**4] for n in ns], float)
    y = np.array([E0[n]/n for n in ns])
    return np.linalg.solve(A, y)[0]
print('3-pt (24,26,28):', f'{solve3((24,26,28)):.14f}', 'dev', f'{solve3((24,26,28)) - target:+.3e}')
