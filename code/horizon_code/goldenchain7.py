# Analytic check: golden chain = TL chain (d = 2cos(pi/5)), which maps to XXZ.
# H_TL = -sum_i P_i = -(1/phi) sum_i e_i.
# Spin-1/2 TL representation: e_i = (rotated) XXZ bond + const:
#   -e_i/phi -> (SxSx+SySy) + cos(g) SzSz - cos(g)/4 ... check numerically first
#   at small N with OPEN boundaries (rep is exact there), then use the exact
#   Bethe integral for the XXZ gs density.
import numpy as np
from scipy.integrate import quad
PHI = (1 + 5 ** 0.5) / 2
g = np.pi / 5   # d = 2 cos g = phi

# --- exact XXZ gs energy density, H = sum (SxSx + SySy + Delta SzSz), Delta = cos(mu), 0<mu<pi
def e_xxz(mu):
    I, err = quad(lambda x: np.sinh((np.pi - mu) * x) / (np.sinh(np.pi * x) * np.cosh(mu * x)),
                  0, 60, limit=500)
    return np.cos(mu) / 4 - np.sin(mu) * I   # integral doubled: 2 * (1/2) * int_0^inf
# sanity: Delta = 0 -> -1/pi
print('sanity e_xxz(pi/2) =', f'{e_xxz(np.pi/2):.10f}', ' vs -1/pi =', f'{-1/np.pi:.10f}')
# sanity: Delta -> 1 (Heisenberg) -> 1/4 - ln 2
print('sanity e_xxz(->0)  =', f'{e_xxz(1e-6):.10f}', ' vs 1/4 - ln2 =', f'{0.25 - np.log(2):.10f}')

# --- TL as XXZ: spectrum of sum e_i (open chain, spin rep) equals
#     sum_i [ -(SxSx+SySy) + cos g SzSz ] + (N-1) cos g /4 ... verify numerically
def tl_spin_rep(N):
    # e_i on spins: e = |s><s| projector-ish: standard: e_i acts on sites i,i+1:
    # in basis up/down: e = [[0,0,0,0],[0,q,-1? ...]] use matrix: e = 
    # [[0,0,0,0],[0,x,y,0],[0,y,1/x? ...]]; canonical TL rep:
    # e_i = [[0,0,0,0],[0,q^{-1},-1,0]? ... use: e = u u^T with u = (0, q^{1/2}, -q^{-1/2}, 0)/norm...
    q = np.exp(1j * g)
    u = np.array([0, q ** 0.5, -q ** -0.5, 0], complex)
    e_loc = np.outer(u, u.conj())   # this satisfies e^2 = d e with d = |u|^2 = q + 1/q = 2cos g
    H = np.zeros((2 ** N, 2 ** N), complex)
    for i in range(N - 1):
        op = np.eye(1)
        for k in range(N - 1):
            op = np.kron(op, e_loc if k == i else np.eye(2)) if False else op
        # build properly
        left = np.eye(2 ** i)
        right = np.eye(2 ** (N - i - 2))
        H += np.kron(np.kron(left, e_loc), right)
    return H

def tl_path_open(N):
    # open golden chain: N anyons, N-1 P operators; path basis x_0..x_N with x_0 = 0 (vacuum start), 
    # steps: x_{k+1} in fusion of x_k with tau. Use links x_1..x_{N-1} internal; simpler: 
    # strings over {0,1} length N+1, x0=0, xN in {0,1} free, adjacency: no two 0s adjacent
    states = []
    def rec(pref):
        if len(pref) == N + 1: states.append(tuple(pref)); return
        for nxt in (0, 1):
            if pref and pref[-1] == 0 and nxt == 0: continue
            rec(pref + [nxt])
    rec([0])
    idx = {s: j for j, s in enumerate(states)}
    D = len(states)
    H = np.zeros((D, D))
    p00, p11, p01 = 1 / PHI ** 2, 1 / PHI, PHI ** -1.5
    for j, s in enumerate(states):
        for i in range(1, N):     # P acts on internal links 1..N-1
            a, x, c = s[i - 1], s[i], s[i + 1]
            if a != c: continue
            if a == 0: H[j, j] += 1.0
            else:
                H[j, j] += p11 if x else p00
                s2 = list(s); s2[i] = 1 - x
                H[idx[tuple(s2)], j] += p01
    return H

for N in [6, 8]:
    Hs = tl_spin_rep(N)             # sum of TL e_i, spin rep
    ws = np.linalg.eigvalsh(Hs)
    Hp = tl_path_open(N) * PHI      # sum of e_i = phi * sum P_i, path rep
    wp = np.linalg.eigvalsh(Hp)
    print(f'N={N}: max eigenvalue sum-e spin rep = {ws[-1]:.10f}   path rep = {wp[-1]:.10f}  (should agree)')

# --- convert: H_golden = -(1/phi) sum e_i. Spin rep: e_loc as matrix:
q = np.exp(1j * g)
u = np.array([0, q ** 0.5, -q ** -0.5, 0], complex)
e_loc = np.outer(u, u.conj())
print('\ne_loc real form:\n', np.round(e_loc.real, 6), '\nimag:\n', np.round(e_loc.imag, 6))
# e_loc = [[0...],[0, e^{?}...]] decompose into spin ops:
# e = cos g/2 *? Let's extract: coefficient of (SxSx+SySy) is -(off diag)= 1 (with sign), SzSz coeff etc.
# generic 2-site op: a) diag(0, A, B, 0) + offdiag C on (01),(10).
A = e_loc[1, 1].real; B = e_loc[2, 2].real; C = e_loc[1, 2]
print('A =', A, 'B =', B, 'C =', C)
# SxSx+SySy has offdiag 1/2 on (01),(10). SzSz = diag(1/4,-1/4,-1/4,1/4).
# e = alpha (SxSx+SySy) + beta SzSz + gamma (Sz_i - Sz_j) + delta I:
# offdiag: alpha/2 = C -> alpha = 2C (complex -> the imaginary part is the boundary term)
# diag: A = -beta/4 + gamma*0? (Sz_i - Sz_j on |01> = (1/2 - (-1/2))?? |01>: Sz_i=1/2? depends ordering)
# not needed in detail: bulk energy density only needs alpha, beta, delta.
alpha = 2 * C.real; beta = -2 * (A + B) / 2 * 2  # solve: A = alpha*0 + beta*(-1/4) + gamma*(1) + delta ...
# do it properly with lstsq over the 4 diag + off-diag equations:
# ops: I, SzSz, Sz_i, Sz_j, (SxSx+SySy)
M = []
targ = []
diagI  = [1, 1, 1, 1]
diagZZ = [0.25, -0.25, -0.25, 0.25]
diagZi = [0.5, 0.5, -0.5, -0.5]
diagZj = [0.5, -0.5, 0.5, -0.5]
for k in range(4):
    M.append([diagI[k], diagZZ[k], diagZi[k], diagZj[k]])
    targ.append(e_loc[k, k].real)
sol, *_ = np.linalg.lstsq(np.array(M), np.array(targ), rcond=None)
dI, dZZ, dZi, dZj = sol
print('e_loc = %.6f I + %.6f SzSz + %.6f Sz_i + %.6f Sz_j + (%.6f + %.6fi)(hopping)' %
      (dI, dZZ, dZi, dZj, C.real, C.imag))
# bulk: sum_i e_i ~ sum [ dI + dZZ SzSz + (dZi + dZj) Sz + 2*Re(C) (SxSx+SySy)/?? ]
# hopping coeff: (SxSx+SySy) contributes 1/2 to offdiag; so coefficient = 2C (complex).
# The imaginary part sums to a boundary term (telescopes) -> drop in bulk.
# In gs sector <Sz> = 0. So:
#   density of sum e_i = dI + dZZ * <SzSz> + 2 Re(C) * <SxSx + SySy>
# and XXZ with H' = sum [ (SxSx+SySy) + Delta' SzSz ]: match Delta' = dZZ / (2 Re C) after scaling.
scale = 2 * C.real     # coefficient in front of (SxSx+SySy)
DeltaP = dZZ / scale
print('scale =', scale, ' effective Delta =', DeltaP, ' (-cos g =', -np.cos(g), ')')
mu = np.arccos(DeltaP)
ex = e_xxz(mu)
eps_sum_e = dI + scale * ex        # density of sum e_i
eps_golden = -(1 / PHI) * eps_sum_e
print('\nanalytic golden-chain density = -(1/phi)*(dI + scale*e_xxz) =', f'{eps_golden:.12f}')
print('numerical extrapolation       = -0.763932018')
print('sqrt(5) - 3                   =', f'{5**0.5 - 3:.12f}')
