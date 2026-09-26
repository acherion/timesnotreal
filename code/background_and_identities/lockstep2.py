# Follow-up precision pass on the lockstep verdict.
# 1. H*t golden crossings: exact times, precision window at today's residual.
# 2. Comoving box vs comoving Hubble radius: crossings, minimum.
# 3. Containment cost at matter re-domination; expansion factors (a at marks).
import math

PHI = (1 + 5**0.5) / 2
TARGET = 4 / PHI**3            # 0.944272
C = 0.133
Om0, OD0 = 0.312, 0.688
H0 = 0.0689
T0 = 0.939 / H0

def deriv(a, rD):
    E = math.sqrt(Om0 / a**3 + rD)
    return H0 * E * a, -3 * C * H0 * rD

a, rD, t, dt = 1.0, OD0, T0, 0.002
prev_Ht, prev_aE = 0.939, 1.0
cross_up = cross_dn = None
win_lo = win_hi = None          # |Ht - target| <= today's residual 0.00527
resid = TARGET - 0.939
box_cross2 = None               # comoving Hubble radius back up through box (aE=1)
min_cHR = (1e9, 0)              # min of 1/(aE), epoch
redom = None
marks = {}
while t < 2500:
    k1 = deriv(a, rD); k2 = deriv(a + 0.5*dt*k1[0], rD + 0.5*dt*k1[1])
    k3 = deriv(a + 0.5*dt*k2[0], rD + 0.5*dt*k2[1]); k4 = deriv(a + dt*k3[0], rD + dt*k3[1])
    a += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    rD += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    t += dt
    E = math.sqrt(Om0 / a**3 + rD)
    Ht = H0 * E * t
    aE = a * E                   # comoving Hubble radius = 1/(aE) in c/H0 units
    OD = rD / (Om0 / a**3 + rD)
    if cross_up is None and prev_Ht < TARGET <= Ht: cross_up = t
    if cross_dn is None and prev_Ht > TARGET >= Ht and t > 20: cross_dn = t
    if abs(Ht - TARGET) <= resid:
        if win_lo is None: win_lo = t
        win_hi = t
    if box_cross2 is None and prev_aE > 1.0 >= aE and t > 20: box_cross2 = t
    if 1 / aE < min_cHR[0]: min_cHR = (1 / aE, t)
    if redom is None and OD < 0.5: redom = (t, 1 / E, a)
    for m in (500, 2000):
        if m not in marks and t >= m: marks[m] = (a, 1 / E, Ht)
    prev_Ht, prev_aE = Ht, aE
    if t > 60 and dt < 0.05: dt = 0.05   # coarsen once past the interesting era

print(f'H*t = 4/phi^3 crossings: UP at t = {cross_up:.2f} Gyr '
      f'({cross_up-T0:.2f} Gyr from today); DOWN at t = {cross_dn:.1f} Gyr; never again')
print(f'precision window (identity as good as today, +-{resid:.5f}): '
      f't = {win_lo:.2f} to {win_hi:.2f} Gyr  -> width {win_hi-win_lo:.2f} Gyr')
print(f'comoving Hubble radius vs box: equal today (crossing), dips to '
      f'{min_cHR[0]:.3f} x box at t = {min_cHR[1]:.0f} Gyr, back through box at t = {box_cross2:.0f} Gyr')
print(f'matter re-domination at t = {redom[0]:.0f} Gyr: containment there would need '
      f'M/M0 = {redom[1]:.0f}; a = {redom[2]:.1f}')
for m in sorted(marks):
    aa, MM, hh = marks[m]
    print(f'  t = {m} Gyr: a = {aa:.0f}, lockstep M/M0 = {MM:.0f}, H*t = {hh:.3f}')
