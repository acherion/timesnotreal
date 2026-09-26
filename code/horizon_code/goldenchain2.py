# Golden chain, pass 2:
#  (a) Temperley-Lieb certificate: e_i = phi * P_i satisfy e^2 = phi e,
#      e_i e_{i+-1} e_i = e_i  -> chain is the integrable A4 RSOS model.
#  (b) AFM towers with correct Delta1 = 3/40 (TIM spin field).
#  (c) FM analysis with mod-3-consistent fits + N=16 tower vs Potts.
#  (d) N=7 cell spectrum located against the even-N TIM tower.
import numpy as np
PHI = (1 + 5 ** 0.5) / 2

def basis(N):
    out = []
    for m in range(1 << N):
        s = [(m >> i) & 1 for i in range(N)]
        if all(s[i] or s[(i + 1) % N] for i in range(N)):
            out.append(tuple(s))
    return out

def P_site(N, i, B=None, idx=None):
    if B is None:
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

# (a) TL certificate at N=7
N0 = 7
B7 = basis(N0); idx7 = {s: j for j, s in enumerate(B7)}
e = [PHI * P_site(N0, i, B7, idx7) for i in range(N0)]
ok_sq  = all(np.allclose(ei @ ei, PHI * ei, atol=1e-12) for ei in e)
ok_eee = all(np.allclose(e[i] @ e[(i+1)%N0] @ e[i], e[i], atol=1e-12) for i in range(N0))
ok_far = np.allclose(e[0] @ e[3], e[3] @ e[0], atol=1e-12)
print('TL certificate: e^2 = phi*e:', ok_sq, ' | e_i e_{i+1} e_i = e_i:', ok_eee, ' | distant commute:', ok_far)
print('-> golden chain = Temperley-Lieb chain at loop weight d = phi = 2cos(pi/5): integrable A4 RSOS')

def spectrum(N, sign):
    B = basis(N); idx = {s: j for j, s in enumerate(B)}
    H = sum(P_site(N, i, B, idx) for i in range(N))
    if sign == 'AFM': return np.linalg.eigvalsh(-H)
    return np.linalg.eigvalsh(H - np.eye(len(B)) * N)

def tower(ev, nmax=14):
    lv = []
    for x in ev[:60]:
        if lv and abs(x - lv[-1][0]) < 1e-8: lv[-1][1] += 1
        else: lv.append([x, 1])
    g = lv[1][0] - lv[0][0]
    return [( (x - lv[0][0]) / g, d) for x, d in lv[:nmax]], g, lv[0][0]

# (b) AFM: r -> c/(12*Delta1) with Delta1 = 3/40 -> 7/9
print('\n=== AFM: c extraction with Delta1 = 3/40 ===')
evs = {N: spectrum(N, 'AFM') for N in [8, 10, 12, 14, 16]}
Ns = np.array(sorted(evs))
A = np.vstack([np.ones_like(Ns, float), 1.0 / Ns.astype(float) ** 2]).T
coef, *_ = np.linalg.lstsq(A, np.array([evs[N][0] for N in Ns]) / Ns, rcond=None)
eps = coef[0]
for N in Ns:
    ev = evs[N]
    gap = next(x - ev[0] for x in ev if x - ev[0] > 1e-9)
    r = (eps * N - ev[0]) / gap
    print(f'N={N:<3} r = {r:.6f}   c = 12*(3/40)*r = {0.9 * r:.6f}')
print('target: r = 7/9 =', f'{7/9:.6f}', '  c = 7/10')

print('\nAFM N=16 tower vs TIM (Delta/(3/40)):')
tw, g16, e016 = tower(evs[16])
tim = {'1(3/40)': 1, '(1/5)': 8/3, '(7/8)': 35/3, '(3/40+1)': 43/3, '(6/5),(1/5+1)': 16, '(7/8+1)': 25, '(3/40+2)': 83/3}
print('observed :', '  '.join(f'{x:.3f}(x{d})' for x, d in tw))
print('TIM      :', '  '.join(f'{k}={v:.3f}' for k, v in tim.items()))

# (c) FM with mod-3-consistent 2-point fits
print('\n=== FM: mod-3-consistent analysis ===')
evF = {N: spectrum(N, 'FM') for N in [8, 10, 12, 14, 16]}
for pair in [(8, 14), (10, 16)]:
    N1, N2 = pair
    # solve E0/N = eps + b/N^2 exactly through two points
    a1, a2 = evF[N1][0] / N1, evF[N2][0] / N2
    b = (a1 - a2) / (1 / N1 ** 2 - 1 / N2 ** 2)
    epsF = a1 - b / N1 ** 2
    for N in pair:
        ev = evF[N]
        gap = next(x - ev[0] for x in ev if x - ev[0] > 1e-9)
        r = (epsF * N - ev[0]) / gap
        print(f'N={N:<3} (pair {pair}, N mod 3 = {N%3}) eps = {epsF:.6f}  r = {r:.6f}  c(Delta1=2/15) = {1.6 * r:.4f}  c(Delta1=1/15) = {0.8 * r:.4f}')
print('Potts targets: Delta1=2/15 -> r = 1/2; if lowest is chiral (1/15,0), Delta=1/15 -> r = 1')
print('\nFM N=16 tower (E-E0)/gap:')
twF, gF, _ = tower(evF[16])
print('observed :', '  '.join(f'{x:.3f}(x{d})' for x, d in twF))
print('Potts Delta ratios if Delta1=2/15: 1, 3 (2/5), 6 (4/5), 8.5? (17/15=2/15+1), 10 (4/3), ...')

# (d) N=7 located against TIM: rescale (E-E0)*N/(2 pi v) using v from even-N gap
# v estimate: gap(N)*N/(2 pi Delta1) with Delta1=3/40, from N=16
v = g16 * 16 / (2 * np.pi * (3 / 40))
print('\n=== N = 7 cell, AFM, scaling dimensions via v =', f'{v:.5f}', '===')
ev7 = spectrum(7, 'AFM')
lv = []
for x in ev7:
    if lv and abs(x - lv[-1][0]) < 1e-8: lv[-1][1] += 1
    else: lv.append([x, 1])
print('E0(7) =', f'{lv[0][0]:.6f}', '  E0/N =', f'{lv[0][0]/7:.6f}', ' vs eps_inf =', f'{eps:.6f}')
print('Delta_eff = (E-E0)*7/(2 pi v):')
for x, d in lv[:8]:
    print(f'   {(x - lv[0][0]) * 7 / (2 * np.pi * v):.4f}  (deg {d})')
print('TIM Deltas: 3/40=0.075, 1/5=0.2, 7/16+7/16=0.875, 43/40=1.075, 6/5=1.2, 3/2? spin fields at odd N shift')
