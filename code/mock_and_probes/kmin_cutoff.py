# CMB low-ell suppression from a primordial cutoff at the parent scale.
# Sachs-Wolfe approximation: C_ell ~ int_kmin dk/k k^(ns-1) j_ell^2(k r_LS)
# Suppression S_ell = C_ell(kmin)/C_ell(0). Compare with the observed low
# quadrupole (Planck D2 ~ 226 muK^2 vs LCDM expectation ~ 1017 muK^2).
import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import quad

ns = 1 - (1+5**0.5)/2 ** 0  # placeholder replaced below
PHI = (1+5**0.5)/2
ns = 1 - PHI**-7            # model's tilt 0.96556
r_LS = 13.9e3               # comoving distance to last scattering, Mpc

def C_ell(ell, kmin):
    # integrate in x = k * r_LS
    xmin = kmin * r_LS
    f = lambda x: x**(ns-2) * spherical_jn(ell, x)**2
    val, err = quad(f, max(xmin, 1e-8), 400, limit=2000)
    return val

kmin_model = 2.29e-4        # 1/r_S(parent) in Mpc^-1  (naive: r_S = 4.37 Gpc)
print('parent-scale cutoff k_min =', kmin_model, 'Mpc^-1;  x_min = k_min*r_LS =', f'{kmin_model*r_LS:.2f}')
print('\nSW suppression factors S_ell = C_ell(cut)/C_ell(0):')
print(' ell   S_ell(model kmin)   S_ell(kmin=3.4e-4)  S_ell(kmin=4.3e-4)')
for ell in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
    c0 = C_ell(ell, 0)
    s1 = C_ell(ell, kmin_model)/c0
    s2 = C_ell(ell, 3.4e-4)/c0
    s3 = C_ell(ell, 4.3e-4)/c0
    print(f'  {ell:<4} {s1:.3f}               {s2:.3f}               {s3:.3f}')

# observed deficit: Planck TT D_2 ~ 226 muK^2 vs LCDM ~ 1017 muK^2 -> ratio ~0.22
print('\nobserved quadrupole ratio D2_obs/D2_LCDM ~ 226/1017 =', f'{226/1017:.3f}')
print('(cosmic variance at ell=2 is ~63%, so ratios 0.2-0.5 are what a cutoff must produce)')

# invert: what k_min best reproduces a quadrupole ratio ~0.22, 0.4?
from scipy.optimize import brentq
c0 = C_ell(2, 0)
for target in [0.22, 0.4, 0.6]:
    kfit = brentq(lambda k: C_ell(2, k)/c0 - target, 1e-5, 2e-3)
    rS = 1/kfit  # Mpc
    M = (2.99792458e8)**2 * (rS*3.0856775814913673e22) / (2*6.6743e-11)
    print(f'quadrupole ratio {target}: k_min = {kfit:.2e} Mpc^-1 -> r = {rS/1000:.2f} Gpc -> M = {M:.2e} kg')
print('\nmodel parent: M = 9.08e52 kg, r_S = 4.37 Gpc, k_min = 2.29e-4')
