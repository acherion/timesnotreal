# The cosmological constant problem, framework decomposition.
# Claim to verify: Lambda * t_Pl^2 = [3 * Omega_DE * (H0 t0)^2 * phi^6 / 64] * (M_Pl / M_parent)^2
# where M_parent = phi^3/8 * c^3 t0 / G  (inverting t0 = (8/phi^3) GM/c^3),
# i.e. the observed 10^-122 is a golden O(1) number times the parent mass squared, inverted.
import numpy as np
PHI = (1 + 5**0.5)/2
G, c, hbar = 6.67430e-11, 2.99792458e8, 1.054571817e-34
t_Pl = np.sqrt(hbar*G/c**5)
M_Pl = np.sqrt(hbar*c/G)
GYR = 3.1556952e16

# observed Lambda (Planck 2018): Lambda = 3 * Omega_L * H0^2 / c^2 ... in m^-2, or as 1/s^2 via 3 Om H0^2
H0_obs = 67.36 * 1e3 / 3.0856775814913673e22    # 1/s
OmL_obs = 0.6889
Lam_obs = 3 * OmL_obs * H0_obs**2               # 1/s^2 (Lambda c^2 convention folded)
print('observed Lambda*t_Pl^2      =', f'{Lam_obs * t_Pl**2:.4e}')
print('the famous ratio: rho_Pl/rho_Lam ~', f'{1/(Lam_obs*t_Pl**2):.3e}', '(the "10^122 problem")')

# model side
Ode = 0.6882                 # (1-p)^n0, derived
H0t0 = 0.9373                # model integrated Friedmann (recomputed earlier tonight)
t0 = 13.4578 * GYR           # model t0 for H0=68.1
Lam_model = 3 * Ode * (H0t0/t0)**2
print('\nmodel Lambda*t0^2           =', f'{3*Ode*H0t0**2:.4f}', ' (doc quotes ~1.82 vs obs 1.869)')

# parent mass from t0 = (8/phi^3) G M / c^3
M_parent = PHI**3/8 * c**3 * t0 / G
print('parent mass M               =', f'{M_parent:.3e} kg  =', f'{M_parent/1.989e30:.3e} Msun')
print('M / M_Pl                    =', f'{M_parent/M_Pl:.3e}')

# the decomposition
golden_prefactor = 3 * Ode * H0t0**2 * PHI**6 / 64
lam_decomp = golden_prefactor * (M_Pl/M_parent)**2
print('\ngolden prefactor 3*Ode*(H0t0)^2*phi^6/64 =', f'{golden_prefactor:.4f}')
print('decomposed Lambda*t_Pl^2    =', f'{lam_decomp:.4e}')
print('model direct  Lambda*t_Pl^2 =', f'{Lam_model*t_Pl**2:.4e}')
print('observed      Lambda*t_Pl^2 =', f'{Lam_obs*t_Pl**2:.4e}')
print('decomposition self-consistent:', np.isclose(lam_decomp, Lam_model*t_Pl**2, rtol=1e-10))
print('model vs observed:', f'{lam_decomp/(Lam_obs*t_Pl**2):.4f}')

# sanity: is the parent an astrophysically unremarkable object *in its own frame*?
# ultramassive BHs in OUR universe: TON 618 ~ 4e10 Msun = 8e40 kg; Phoenix A ~ 1e11 Msun
print('\nparent mass in solar masses:', f'{M_parent/1.989e30:.2e}',
      ' (vs largest known BH in our frame ~1e11 Msun -> ratio', f'{M_parent/1.989e30/1e11:.1e})')
# what Lambda would a TON-618-mass parent give its child?
for name, Msun in [('TON 618 (4e10 Msun)', 4e10), ('Sgr A* (4.3e6 Msun)', 4.3e6), ('10 Msun stellar', 10)]:
    M = Msun * 1.989e30
    lam = golden_prefactor * (M_Pl/M)**2
    print(f'child of {name:<22}: Lambda*t_Pl^2 = {lam:.2e}')
