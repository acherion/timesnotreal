# Does the "fuel processed into matter" narrative survive as literal energy transfer?
# Compare three backgrounds, all with Om0=0.312, ODE0=0.688, and the same DE decline
# rate d ln rho_DE / dt = -3*C*H0 (C=0.133):
#   (A) committed: DE decline is expansion work (w>-1 fluid), matter standard a^-3
#   (B) literal transfer: DE has w=-1, its decline is energy transferred into matter
#   (C) 50/50 split
# Report H(z) and D_M(z) differences vs (A) at BAO redshifts; DESI shape precision ~1%.
import numpy as np
from scipy.integrate import solve_ivp

C0 = 0.133
Om0, OD0 = 0.312, 0.688

def run(frac_transfer):
    # integrate backward in time from today: variables rho_m, rho_DE in units of rho_crit0; t in 1/H0
    def rhs(t, y):
        rm, rd = y
        H = np.sqrt(rm + rd)
        drain = 3*C0*rd                      # total DE decline rate (per 1/H0)
        # DE: expansion-work part uses w; transfer part is Q
        # d rho_DE/dt = -drain  (by construction, same in all variants)
        # d rho_m/dt  = -3 H rho_m + frac*drain
        return [-3*H*rm + frac_transfer*drain, -drain]
    # go back to z=3: need a(t). integrate in ln a instead: d/dlna = (1/H) d/dt
    def rhs_a(lna, y):
        rm, rd = y
        H = np.sqrt(rm + rd)
        drain = 3*C0*rd
        return [(-3*H*rm + frac_transfer*drain)/H, -drain/H]
    sol = solve_ivp(rhs_a, [0, -np.log(1+3.0)], [Om0, OD0], dense_output=True, rtol=1e-9, atol=1e-12)
    zs = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.33])
    E = []
    for z in zs:
        rm, rd = sol.sol(-np.log(1+z))
        E.append(np.sqrt(rm + rd))
    E = np.array(E)
    # comoving distance via fine grid
    zg = np.linspace(0, 3.0, 3001)
    Eg = np.array([np.sqrt(np.sum(sol.sol(-np.log(1+z)))) for z in zg])
    chi = np.concatenate(([0], np.cumsum(0.5*(1/Eg[1:] + 1/Eg[:-1])*np.diff(zg))))
    DM = np.interp(zs, zg, chi)
    rm_z1 = sol.sol(-np.log(2.0))[0]
    return zs, E, DM, rm_z1

zs, EA, DMA, rmA = run(0.0)
for frac, label in ((1.0, 'literal transfer (100% of drain into matter)'), (0.5, '50% transfer')):
    _, EB, DMB, rmB = run(frac)
    print(f'{label}:')
    print('   z     dH/H      dD_M/D_M   (vs committed w-fluid)')
    for z, a, b, c, d in zip(zs, EA, EB, DMA, DMB):
        print(f'  {z:4.2f}  {100*(b/a-1):+6.2f}%   {100*(d/c-1):+6.2f}%')
    print(f'   matter density at z=1: {100*(rmB/rmA-1):+.1f}% vs standard dilution')
    print()
print('DESI DR2 BAO shape precision: ~0.5-1.5% per point; a coherent >3% shift is excluded.')
