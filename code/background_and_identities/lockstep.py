# Post-completion lockstep analysis, three parts.
#
# (A) PEAK THEOREM TEST: under the model's w(z) = -1 + C/E (C = 0.133,
#     exponential DE decay), H*t rises, peaks, and falls to 2/3 (matter era).
#     The age identity requires H*t = 4/phi^3 = 0.94427. So the identity can
#     only hold near the peak. Questions: (i) does max(H*t) equal 4/phi^3
#     exactly? (ii) what C* would pin the peak to 4/phi^3, vs the recursion's
#     C = n0*|ln(1-p)|/(3*H0*t0)? (iii) how wide is the window where the
#     identity holds as well as it does today?
#
# (B) ETERNAL LOCKSTEP DEMAND: if M = c^3/(2GH) forever, M/M0 = 1/E(t).
#     Table of the mass the parent must reach. (Unbounded => dead on arrival
#     if any ceiling exists.)
#
# (C) DRAIN-FED RELEASE: if the parent's growth is only the drain-driven part
#     of the demand, dM/dt = (3c^3/4G) * C*Omega_DE/E, then M saturates:
#     M_inf/M0 = 1 + (3C/2) * integral Omega_DE/E d(H0 t).  Compute M_inf,
#     the escape epoch t_esc where c/H catches r_S(M_inf), and check M_inf
#     against simple golden forms.
import math

PHI = (1 + 5**0.5) / 2
TARGET = 4 / PHI**3
Om0, OD0 = 0.312, 0.688
H0 = 0.0689                     # 1/Gyr
T0 = 0.939 / H0                 # 13.63 Gyr (model's own t0)

def evolve(C, t_end, dt=0.01, collect=False):
    """Integrate (a, rho_DE) forward from today. Returns peak info and
    optional trajectory samples."""
    def deriv(a, rD):
        E = math.sqrt(Om0 / a**3 + rD)
        return H0 * E * a, -3 * C * H0 * rD
    a, rD, t = 1.0, OD0, T0
    peak_val, peak_t = 0.0, 0.0
    window_hi = None            # last t with H*t >= today's 0.939
    traj = []
    integ_drain = 0.0           # integral of Omega_DE/E dt  (for part C)
    while t < t_end:
        k1 = deriv(a, rD)
        k2 = deriv(a + 0.5*dt*k1[0], rD + 0.5*dt*k1[1])
        k3 = deriv(a + 0.5*dt*k2[0], rD + 0.5*dt*k2[1])
        k4 = deriv(a + dt*k3[0], rD + dt*k3[1])
        a += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        rD += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        t += dt
        E = math.sqrt(Om0 / a**3 + rD)
        Ht = H0 * E * t
        OD = rD / (Om0 / a**3 + rD)
        integ_drain += (OD / E) * dt
        if Ht > peak_val:
            peak_val, peak_t = Ht, t
        if Ht >= 0.939:
            window_hi = t
        if collect:
            traj.append((t, a, E, OD, Ht))
    return peak_val, peak_t, window_hi, integ_drain, traj

# ---------- (A) peak theorem ----------
C_REC = 0.133
pv, pt, whi, _, _ = evolve(C_REC, 80, dt=0.002)
print('(A) PEAK OF H*t UNDER THE MODEL\'S OWN w(z)')
print(f'    4/phi^3 target        = {TARGET:.6f}')
print(f'    max(H*t)              = {pv:.6f}   at t = {pt:.2f} Gyr  (t0 = {T0:.2f})')
print(f'    miss                  = {(pv/TARGET-1)*100:+.4f}%')
print(f'    identity window (H*t >= today\'s 0.939): t = {T0:.1f} to {whi:.1f} Gyr '
      f'({whi-T0:.1f} Gyr long)')

# sensitivity and fixed point: what C pins the peak exactly on 4/phi^3?
print('\n    sensitivity of the peak to the drain constant C:')
for C in (0.100, 0.120, 0.133, 0.146, 0.170):
    v, tt, _, _, _ = evolve(C, 80, dt=0.005)
    print(f'      C = {C:.3f}:  max(H*t) = {v:.5f}  ({(v/TARGET-1)*100:+.2f}% from 4/phi^3)')
lo, hi = 0.09, 0.20
for _ in range(40):
    mid = 0.5 * (lo + hi)
    v, _, _, _, _ = evolve(mid, 80, dt=0.005)
    if v > TARGET: lo, hi = lo, mid     # larger C -> DE dies faster -> lower peak
    else: lo, hi = mid, hi
Cstar = 0.5 * (lo + hi)
n0, p = 2 * PHI**2, 2 / PHI**7
C_from_recursion = n0 * abs(math.log(1 - p)) / (3 * 0.939)
print(f'\n    C* that pins the peak exactly at 4/phi^3: {Cstar:.5f}')
print(f'    C from the recursion n0|ln(1-p)|/(3 H0t0):  {C_from_recursion:.5f}'
      f'   (ratio {Cstar/C_from_recursion:.4f})')

# ---------- (B) eternal lockstep demand ----------
_, _, _, integ, traj = evolve(C_REC, 3000, dt=0.01, collect=True)
print('\n(B) ETERNAL LOCKSTEP: mass the parent must reach (M/M0 = 1/E)')
marks = {20: None, 50: None, 100: None, 500: None, 2000: None}
t2 = t10 = None
for (t, a, E, OD, Ht) in traj:
    for m in marks:
        if marks[m] is None and t >= m: marks[m] = 1 / E
    if t2 is None and 1 / E >= 2: t2 = t
    if t10 is None and 1 / E >= 10: t10 = t
for m in sorted(marks):
    print(f'      t = {m:5d} Gyr:  M/M0 = {marks[m]:8.2f}')
print(f'    M doubles by t = {t2:.0f} Gyr; 10x by t = {t10:.0f} Gyr; unbounded (M ~ t).')

# ---------- (C) drain-fed release ----------
Minf = 1 + 1.5 * C_REC * H0 * integ
print('\n(C) DRAIN-FED PARENT: M grows only by the drain-driven demand')
print(f'    M_inf/M0 = 1 + (3C/2) integral[Omega_DE/E d(H0 t)] = {Minf:.5f}')
cands = [('1 + 2/phi^7 (=1+p)', 1 + 2/PHI**7), ('1 + 1/phi^5', 1 + PHI**-5),
         ('1 + 1/phi^4', 1 + PHI**-4), ('1 + 1/phi^3', 1 + PHI**-3),
         ('1 + C', 1 + C_REC), ('phi^(1/4)', PHI**0.25),
         ('1 + 1/(2 phi^3)', 1 + 0.5*PHI**-3), ('1 + p*phi (=1+2/phi^6)', 1 + 2/PHI**6)]
best = sorted(cands, key=lambda c: abs(c[1]/Minf - 1))[:3]
for name, val in best:
    print(f'      vs {name:<22} = {val:.5f}   ({(val/Minf-1)*100:+.2f}%)')
tesc = None
for (t, a, E, OD, Ht) in traj:
    if 1 / E >= Minf: tesc = t; break
print(f'    containment ends (c/H catches r_S(M_inf)) at t_esc = {tesc:.0f} Gyr'
      f'  =  {tesc/T0:.1f} x t0')

# bookkeeping today: internal DE pressure-work supply vs lockstep demand
wtot = (-1 + C_REC) * OD0
supply_demand = -2 * wtot / (1 + wtot)
print(f'\n    today: DE pressure-work supply / lockstep demand = {supply_demand:.2f}'
      f'  (DE era over-contains; matter era under-supplies)')
