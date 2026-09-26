# Brute-forcing Q1's computable branches.
# (i)  Is the "milestone" reading internally consistent? (algebra check)
# (ii) Is DE/DM = phi^2 a dynamical attractor, or a transient crossing?
#      -> track the ratio's full trajectory, measure crossing width
# (iii) "Time constant" reading: how wide is the window in which the
#      recursion is DETECTABLE (|1+w|*Omega_DE big enough to measure)?
#      If window >> crossing width, reading (c) does NOT explain the timing.
import math

PHI = (1 + 5**0.5) / 2
C = 0.133
Om0, OD0 = 0.312, 0.688
ODM0 = 0.263                      # dark matter only (baryons excluded)
H0 = 0.0689                       # 1/Gyr
t0 = 0.939 / H0

# --- (i) algebra: H*t at the moment depth = n0, given r_S = r_H ---
# t = n0 * (4/phi^5) * GM/c^3  and  GMH/c^3 = 1/2  =>  H*t = n0*(4/phi^5)/2 = 4/phi^3
n0 = 2 * PHI**2
print(f'(i) H*t forced by [r_S=r_H] + [t = n0 levels]: {n0 * (4/PHI**5) / 2:.4f} '
      f'= 4/phi^3 = {4/PHI**3:.4f}  -> milestone reading is internally consistent')

# --- (ii)+(iii) integrate past and future ---
def deriv(a, rD):
    E = math.sqrt(Om0 / a**3 + rD)
    return H0 * E * a, -3 * C * H0 * rD

# integrate backward to a=0.05 and forward to 400 Gyr from (a=1, rD=OD0)
states = []
a, rD, t, dt = 1.0, OD0, t0, 0.002
while a > 0.03:                    # backward
    k1 = deriv(a, rD); k2 = deriv(a - 0.5*dt*k1[0], rD - 0.5*dt*k1[1])
    k3 = deriv(a - 0.5*dt*k2[0], rD - 0.5*dt*k2[1]); k4 = deriv(a - dt*k3[0], rD - dt*k3[1])
    a -= dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); rD -= dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    t -= dt
    states.append((t, a, rD))
states.reverse()
a, rD, t = 1.0, OD0, t0
states.append((t0, 1.0, OD0))
while t < 400:                     # forward
    k1 = deriv(a, rD); k2 = deriv(a + 0.5*dt*k1[0], rD + 0.5*dt*k1[1])
    k3 = deriv(a + 0.5*dt*k2[0], rD + 0.5*dt*k2[1]); k4 = deriv(a + dt*k3[0], rD + dt*k3[1])
    a += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); rD += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    t += dt
    states.append((t, a, rD))

# ratio trajectory
in10, lo_t, hi_t, peak = None, None, None, (0, 0)
det_lo, det_hi = None, None       # detectability window: Omega_DE*|1+w| > 0.01
for (t, a, rD) in states:
    rDM = ODM0 / a**3
    ratio = rD / rDM
    E = math.sqrt(Om0 / a**3 + rD)
    OD = rD / (Om0 / a**3 + rD)
    sig = OD * (C / E)            # Omega_DE * |1+w|: size of the H(z) signature
    if ratio > peak[1]: peak = (t, ratio)
    if abs(ratio / PHI**2 - 1) < 0.10:
        if lo_t is None: lo_t = t
        hi_t = t
    if sig > 0.01:
        if det_lo is None: det_lo = t
        det_hi = t

print(f'\n(ii) DE/DM ratio trajectory: 0 (early) -> today {OD0/ODM0:.3f} (phi^2 = {PHI**2:.3f})')
print(f'     peak ratio = {peak[1]:.0f} at t = {peak[0]:.0f} Gyr, then back to 0')
print(f'     time spent within +-10% of phi^2: {hi_t - lo_t:.1f} Gyr '
      f'(t = {lo_t:.1f} to {hi_t:.1f}; universe age {t0:.1f})')
print(f'     -> phi^2 is a TRANSIENT CROSSING, not an attractor')

print(f'\n(iii) recursion detectable (Omega_DE*|1+w| > 0.01, i.e. ~1% effect on H):')
print(f'     from t = {det_lo:.1f} Gyr to t = {det_hi:.0f} Gyr  (width {det_hi-det_lo:.0f} Gyr)')
print(f'     crossing window / detectability window = {(hi_t-lo_t)/(det_hi-det_lo)*100:.1f}%')
