# Handles on the parent mass / mass ratio / recursion depth.
# (1) internal consistency: M_parent should equal the Hubble mass (r_S = r_H)
# (2) downhill ratio: parent mass / largest BH our universe has produced
# (3) implied number of generations between us and the Planck floor, vs n0 = 2 phi^2
# (4) CMB low-quadrupole handle: k_min cutoff <-> parent Schwarzschild radius
import numpy as np
PHI = (1+5**0.5)/2
G, c, hbar = 6.67430e-11, 2.99792458e8, 1.054571817e-34
M_Pl = np.sqrt(hbar*c/G)
Msun = 1.989e30
GYR = 3.1556952e16
MPC = 3.0856775814913673e22

t0 = 13.4578*GYR
H0 = 68.1e3/MPC
M_parent = PHI**3/8 * c**3 * t0 / G

# (1) Hubble mass
M_H = c**3/(2*G*H0)
print('(1) M_parent =', f'{M_parent:.3e} kg   Hubble mass c^3/(2G H0) =', f'{M_H:.3e} kg',
      '  ratio', f'{M_parent/M_H:.3f}')
print('    (r_S = r_H: the parent mass IS the mass of our observable universe)')

# (2)+(3) downhill ratio and implied depth to Planck floor
lnMP = np.log(M_parent/M_Pl)
print('\n(2) M_parent/M_Pl =', f'{M_parent/M_Pl:.3e}', ' ln =', f'{lnMP:.2f}', ' log10 =', f'{lnMP/np.log(10):.2f}')
print('(3) generations below us until child BHs hit the Planck floor, if each level')
print('    shrinks by (parent mass)/(largest BH it breeds):')
for name, Ms in [('TON 618        (4.0e10 Msun)', 4.0e10),
                 ('Phoenix A      (1.0e11 Msun)', 1.0e11),
                 ('largest = 2x Phoenix (2e11)  ', 2.0e11),
                 ('largest = 0.5x TON618 (2e10) ', 2.0e10)]:
    Mtop = Ms*Msun
    R = M_parent/Mtop
    depth = lnMP/np.log(R)
    print(f'    {name}: ratio = {R:.2e}  -> depth = {depth:.3f}')
print('    n0 = 2 phi^2 =', f'{2*PHI**2:.4f}')

# what per-level ratio would make the depth EXACTLY n0?
R_exact = np.exp(lnMP/(2*PHI**2))
print('    ratio required for depth = n0 exactly:', f'{R_exact:.3e}',
      '-> top BH =', f'{M_parent/R_exact/Msun:.3e} Msun')

# (4) CMB cutoff: if primordial modes are cut at the parent's r_S = r_H scale
r_S = 2*G*M_parent/c**2
print('\n(4) parent r_S =', f'{r_S/MPC/1e3:.2f} Gpc', ' -> k_min = 1/r_S =',
      f'{MPC*1e3/ (r_S/1):.2e}'.replace('e','e'), end='')
k_min = 1/(r_S/MPC)   # in 1/Mpc
print(f'  k_min = {k_min:.2e} Mpc^-1')
print('    literature k_min fits to the low-quadrupole deficit: ~2-5 x 10^-4 Mpc^-1')
print('    comoving horizon today ~ 14.3 Gpc -> k ~ 7e-5 Mpc^-1 (r_S is the naive scale;')
print('    a proper comparison needs the comoving mapping of the bounce scale)')
