# Q3: Kerr horizon oblateness at the universal spin -> anisotropy of the k_min cutoff.
# Q5: uniqueness audit of the alpha formula's coefficient pattern.
import math
from itertools import combinations, product

PHI = (1 + 5**0.5) / 2

# ---------- Q3: oblateness ----------
a = 0.686
rp = 1 + math.sqrt(1 - a * a)           # r_+ in units M=1
# equatorial circumference: 2*pi*(rp^2+a^2)/rp = 4*pi exactly (horizon identity)
Ce = 2 * math.pi * (rp**2 + a**2) / rp
n = 200000
Cp = 2 * sum(math.sqrt(rp**2 + a**2 * math.cos((i + 0.5) / n * math.pi)**2)
             for i in range(n)) * math.pi / n
print('Q3: Kerr horizon shape at the universal remnant spin a* = 0.686')
print(f'  equatorial circumference = {Ce/math.pi:.4f} pi M (exactly 4 pi M for any Kerr)')
print(f'  polar circumference      = {Cp/math.pi:.4f} pi M')
print(f'  oblateness 1 - Cp/Ce     = {1 - Cp/Ce:.4f}  ({100*(1-Cp/Ce):.1f}%)')
print(f'  -> if box GEOMETRY crosses (as k_min=1/r_S already assumes), a merger-born')
print(f'     parent imprints a ~{100*(1-Cp/Ce):.0f}% direction-dependence on the cutoff:')
print(f'     more low-ell suppression along the spin axis - quadrupolar, one axis,')
print(f'     correlated with the low-power anomaly. Collapse-born parent: none.')

# ---------- Q5: alpha-formula uniqueness audit ----------
# G-anchored target: alpha(m_p) s.t. G = k_e e^2/m_p^2 * alpha^(4 phi^3) exactly
ke, e, mp, G = 8.9875517923e9, 1.602176634e-19, 1.67262192369e-27, 6.67430e-11
R = ke * e * e / (G * mp * mp)          # EM/gravity force ratio for protons
inv_alpha_G = R ** (1 / (4 * PHI**3))
F = (PHI**11 + PHI**9 - 2 * PHI**2) / 2
delta = abs(F / inv_alpha_G - 1)
print(f'\nQ5: target 1/alpha_G = {inv_alpha_G:.4f} (from measured G);  formula = {F:.4f}')
print(f'  formula miss = {delta*100:.4f}%')

# enumerate all (c1 phi^e1 + c2 phi^e2 + c3 phi^e3)/2, e1>e2>e3 in 0..14, |ci|<=3
pw = [PHI**k for k in range(15)]
LATTICE = {0, 2, 4, 6, 7, 9, 11, 13, 14}    # exponents of form 7a+2b, a<=2, b<=3
hits = []
for (e1, e2, e3) in combinations(range(14, -1, -1), 3):
    for (c1, c2, c3) in product(range(-3, 4), repeat=3):
        if c1 == 0: continue                 # leading term nonzero; c2/c3 may vanish
        X = (c1 * pw[e1] + c2 * pw[e2] + c3 * pw[e3]) / 2
        if X <= 0: continue
        if abs(X / inv_alpha_G - 1) <= delta:
            hits.append(((c1, e1), (c2, e2), (c3, e3),
                         abs(c1) + abs(c2) + abs(c3), abs(X / inv_alpha_G - 1)))
print(f'  expressions matching AS WELL OR BETTER than the formula: {len(hits)}')
lat = [h for h in hits if all(t[1] in LATTICE or t[0] == 0 for t in h[:3])]
print(f'  ...restricted to the physical (aN+bd) exponent lattice: {len(lat)}')
lat.sort(key=lambda h: (h[3], -h[4]))
for h in lat[:6]:
    terms = ' + '.join(f'{t[0]}*phi^{t[1]}' for t in h[:3] if t[0] != 0)
    print(f'     [{terms}]/2   (|c| sum {h[3]}, miss {h[4]*100:.4f}%)')
# value-density baseline: how many combos land in a random window of same width?
import random
random.seed(1)
base = 0
for _ in range(20):
    T = random.uniform(120, 150)
    cnt = 0
    for (e1, e2, e3) in combinations(range(14, -1, -1), 3):
        for (c1, c2, c3) in product(range(-3, 4), repeat=3):
            if c1 == 0: continue
            X = (c1 * pw[e1] + c2 * pw[e2] + c3 * pw[e3]) / 2
            if X > 0 and abs(X / T - 1) <= delta:
                cnt += 1
    base += cnt
print(f'  baseline: random target in [120,150] matches {base/20:.1f} expressions on average')
