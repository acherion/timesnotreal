# Q2 brute force, first pass: search DESI DR2 BAO for a STEP in H(z).
# Physics target: if the parent BH swallowed a large object at epoch t*,
# M steps up => H steps down at z*. Phenomenology: E(z) -> E(z)*(1+A) for z>z*.
# Fit smooth model (recursion w(z); LCDM cross-check) + step; scan z*.
import math
import numpy as np
from scipy.optimize import minimize

C = 0.133

# DESI DR2 BAO (tracer, zeff, DV, sDV, DM, sDM, DH, sDH, r_MH)
DATA = [
    ('BGS',  0.295,  7.944, 0.075, None,   None,  None,   None,  None),
    ('LRG1', 0.510, None,   None,  13.587, 0.169, 21.863, 0.427, -0.475),
    ('LRG2', 0.706, None,   None,  17.347, 0.180, 19.458, 0.332, -0.423),
    ('LRG3E',0.934, None,   None,  21.574, 0.153, 17.641, 0.193, -0.425),
    ('ELG2', 1.321, None,   None,  27.605, 0.320, 14.178, 0.217, -0.437),
    ('QSO',  1.484, None,   None,  30.519, 0.758, 12.816, 0.513, -0.489),
    ('Lya',  2.330, None,   None,  38.988, 0.531,  8.632, 0.101, -0.431),
]

ZG = np.linspace(0, 2.6, 2601)

def build_E(Om, recursion=True):
    OD = 1 - Om
    E = np.sqrt(Om * (1 + ZG)**3 + OD)
    if not recursion:
        return E
    for _ in range(40):
        integ = C / E / (1 + ZG)
        I = np.concatenate([[0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(ZG))])
        En = np.sqrt(Om * (1 + ZG)**3 + OD * np.exp(3 * I))
        if np.max(np.abs(En - E)) < 1e-12:
            return En
        E = En
    return E

def chi2(params, zstar, Esm):
    beta, A = params[0], params[1]
    Estep = np.where(ZG > zstar, Esm * (1 + A), Esm) if zstar is not None else Esm
    inv = 1 / Estep
    DMg = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(ZG))])
    tot = 0.0
    for (_, z, DV, sDV, DM, sDM, DH, sDH, r) in DATA:
        i = int(round(z / 2.6 * 2600))
        dm, dh = beta * DMg[i], beta / Estep[i]
        if DV is not None:
            dv = (z * dm * dm * dh) ** (1 / 3)
            tot += ((dv - DV) / sDV) ** 2
        else:
            dM, dH = dm - DM, dh - DH
            det = (sDM * sDH) ** 2 * (1 - r * r)
            tot += (dM**2 * sDH**2 + dH**2 * sDM**2 - 2 * r * sDM * sDH * dM * dH) / det
    return tot

def fit(zstar, Esm, with_step):
    if with_step:
        res = minimize(lambda p: chi2(p, zstar, Esm), [29.0, 0.0],
                       method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8})
        return res.fun, res.x
    res = minimize(lambda p: chi2([p[0], 0.0], None, Esm), [29.0],
                   method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8})
    return res.fun, res.x

for label, rec in [('recursion w(z)', True), ('LCDM', False)]:
    # profile over Om on a grid (cheap and robust)
    best0 = (1e9, None, None)
    for Om in np.arange(0.28, 0.35, 0.002):
        Esm = build_E(Om, rec)
        c0, p0 = fit(None, Esm, False)
        if c0 < best0[0]: best0 = (c0, Om, Esm)
    chi0, Om0, Esm0 = best0
    print(f'--- smooth model: {label} ---')
    print(f'  baseline fit: chi2 = {chi0:.2f} (13 pts, 2 params: beta, Om={Om0:.3f})')
    zbest, dbest, Abest = None, 0, 0
    prof = []
    for zstar in np.arange(0.35, 2.25, 0.05):
        cA, pA = fit(zstar, Esm0, True)
        d = chi0 - cA
        prof.append((zstar, d, pA[1]))
        if d > dbest: zbest, dbest, Abest = zstar, d, pA[1]
    print(f'  best step: z* = {zbest:.2f}, A = {Abest:+.4f}, dchi2 = {dbest:.2f} '
          f'(local {math.sqrt(max(dbest,0)):.1f} sigma; ~38 trials)')
    # 95% band on |A| across z* (dchi2 = 3.84 profile, crude scan in A)
    worst = 0
    for zstar in (0.5, 0.9, 1.3, 1.8):
        Esm = Esm0
        lim = None
        for A in np.arange(0.0, 0.10, 0.001):
            c1r = minimize(lambda p: chi2([p[0], A], zstar, Esm), [29.0],
                           method='Nelder-Mead').fun
            c2r = minimize(lambda p: chi2([p[0], -A], zstar, Esm), [29.0],
                           method='Nelder-Mead').fun
            if min(c1r, c2r) - chi0 > 3.84:
                lim = A; break
        worst = max(worst, lim if lim else 0.10)
        print(f'  95% limit on |step| at z*={zstar}: {"" if lim else ">"}{(lim or 0.10)*100:.1f}%')
    print(f'  => no parent-merger step larger than ~{worst*100:.0f}% anywhere in 0.35<z<2.2')
    print()
