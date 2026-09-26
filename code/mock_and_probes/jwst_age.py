# Age of the universe at JWST redshifts: framework w(z) = -1 + 0.133/E(z) vs Planck LCDM.
# Framework DE density: d(ln rho_DE)/dz = 3(1+w)/(1+z), 1+w = 0.133/E(z),
# solved self-consistently with E^2 = Om(1+z)^3 + Or(1+z)^4 + Ode*g(z), g(0)=1.
# Age: t(z) = (1/H0) * int_z^inf dz'/[(1+z')E(z')].
import numpy as np
from scipy.integrate import solve_ivp, quad

MPC_KM = 3.0856775814913673e19          # km per Mpc
GYR_S  = 3.1556952e16                    # seconds per Gyr

def make_E_lcdm(Om, Or):
    Ode = 1 - Om - Or
    def E(z):
        return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode)
    return E

def make_E_model(Om, Or, c0=0.133):
    Ode = 1 - Om - Or
    # integrate g(z) on a grid; 1+w = c0/E -> dg/dz = 3*c0*g/((1+z)*E)
    zs = np.concatenate([np.linspace(0, 20, 4001), np.geomspace(20.01, 1e8, 4000)])
    def rhs(z, y):
        g = y[0]
        E = np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode*g)
        return [3*c0*g/((1+z)*E)]
    sol = solve_ivp(rhs, [0, zs[-1]], [1.0], t_eval=zs, rtol=1e-10, atol=1e-14, method='RK45')
    gz = sol.y[0]
    def E(z):
        g = np.interp(z, zs, gz)
        return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode*g)
    return E, gz, zs

def age_at(E, z, H0_kms_Mpc):
    H0 = H0_kms_Mpc / MPC_KM            # 1/s
    # substitute u = ln(1+z')
    f = lambda u: 1.0 / E(np.expm1(u))
    val, err = quad(f, np.log(1+z), np.log(1e8), limit=800)
    # radiation-domination tail beyond 1e8: t ~ negligible (~<1e-6 Gyr), add analytic
    return val / H0 / GYR_S

# --- Planck LCDM ---
h_P = 0.6736
Om_P, Or_P = 0.3153, 9.24e-5 / h_P**2 * h_P**2  # Or approx 9.24e-5 at h=0.6736 (photons+3 nu)
Or_P = 9.24e-5
E_P = make_E_lcdm(Om_P, Or_P)

# --- framework ---
h_M = 0.681                    # model H0 = 68.1 (its stated prediction)
Om_M = 1 - 0.6882 - 9.05e-5    # Ode = 0.6882 exact golden value
Or_M = 9.24e-5 * (0.6736/h_M)**2   # same physical radiation density (T_CMB fixed) -> Or h^2 fixed
Om_M = 1 - 0.6882 - Or_M
E_M, gz, zgrid = make_E_model(Om_M, Or_M)

# sanity: model H0*t0 should reproduce ~0.939 (documented value)
t0_M = age_at(E_M, 0, 68.1)
t0_P = age_at(E_P, 0, 67.36)
H0t0_M = 68.1/MPC_KM * t0_M * GYR_S
print(f'sanity: model H0*t0 = {H0t0_M:.4f}  (documented: 0.939)')
print(f'LCDM   t0 = {t0_P:.4f} Gyr (Planck says 13.797)')
print(f'model  t0 = {t0_M:.4f} Gyr')

print('\nage at JWST redshifts (Myr):')
print(' z     LCDM(P18)   model(H0=68.1)  diff(Myr)  diff(%)')
for z in [8, 10, 12, 14, 16, 20, 25, 30]:
    tP = age_at(E_P, z, 67.36) * 1000
    tM = age_at(E_M, z, 68.1) * 1000
    print(f'{z:>3}    {tP:8.2f}     {tM:8.2f}      {tM-tP:+7.2f}   {(tM/tP-1)*100:+.2f}%')

# variant: model with SAME H0 as Planck (in case one argues the anchor)
print('\nvariant: model with H0 = 67.36 (same anchor as Planck):')
for z in [10, 14, 20]:
    tP = age_at(E_P, z, 67.36) * 1000
    tM = age_at(E_M, z, 67.36) * 1000
    print(f'z={z:<3}  LCDM {tP:8.2f}  model {tM:8.2f}   diff {tM-tP:+7.2f} Myr ({(tM/tP-1)*100:+.2f}%)')

# where does the DE difference actually live? w and DE fraction vs z
print('\nmodel DE properties:')
for z in [0, 0.5, 1, 2, 5, 10, 14]:
    E = E_M(z)
    w = -1 + 0.133/E
    g = np.interp(z, zgrid, gz)
    fDE = 0.6882*g/E**2
    print(f'z={z:<5} E={E:9.3f}  w={w:+.4f}  rho_DE/rho_DE0={g:.4f}  Omega_DE(z)={fDE:.4f}')

# growth-of-structure side quick look: linear growth factor ODE
# D'' + [3/a + E'/E ] D' - (3/2) Om a^-3 / E^2 * D/a^2 = 0  in ln a
def growth(E_func, Om):
    # integrate d2D/dlna2 + (2 + dlnE/dlna) dD/dlna = 1.5 * Om a^-3/E^2 D
    lna = np.linspace(np.log(1e-4), 0, 6000)
    def rhs(la, y):
        a = np.exp(la); z = 1/a - 1
        E = E_func(z)
        dla = 1e-4
        E2 = E_func(1/np.exp(la+dla) - 1)
        dlnE = (np.log(E2) - np.log(E)) / dla
        D, Dp = y
        return [Dp, -(2 + dlnE)*Dp + 1.5*Om*(1+z)**3/E**2 * D]
    sol = solve_ivp(rhs, [lna[0], 0], [np.exp(lna[0]), np.exp(lna[0])], t_eval=lna, rtol=1e-9)
    return sol.t, sol.y[0]
laP, DP = growth(E_P, Om_P)
laM, DM = growth(E_M, Om_M)
# sigma8-like ratio today (normalized same early amplitude)
print('\nlinear growth D(z=0)/D(z=1100-ish), normalized at a=1e-4:')
print(f'LCDM : {DP[-1]/DP[0]:.2f}')
print(f'model: {DM[-1]/DM[0]:.2f}   ratio model/LCDM = {(DM[-1]/DM[0])/(DP[-1]/DP[0]):.4f}')
