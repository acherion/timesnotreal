# Verify: f*sigma8(z) for framework w(z) vs Planck LCDM against RSD measurements.
# Growth ODE in ln a: D'' + (2 + dlnE/dlna) D' = 1.5 Om(1+z)^3/E^2 D
# f = dlnD/dlna ; fs8(z) = f(z) * sigma8_0 * D(z)/D(0)
# Normalization: SAME primordial amplitude for both -> sigma8_0(model) = 0.8111 * D_M(0)/D_P(0)
import numpy as np
from scipy.integrate import solve_ivp

def make_E_lcdm(Om, Or):
    Ode = 1 - Om - Or
    return lambda z: np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode)

def make_E_model(Om, Or, c0=0.133):
    Ode = 1 - Om - Or
    zs = np.concatenate([np.linspace(0, 20, 4001), np.geomspace(20.01, 1e6, 3000)])
    def rhs(z, y):
        E = np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode*y[0])
        return [3*c0*y[0]/((1+z)*E)]
    sol = solve_ivp(rhs, [0, zs[-1]], [1.0], t_eval=zs, rtol=1e-10, atol=1e-14)
    gz = sol.y[0]
    return lambda z: np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode*np.interp(z, zs, gz))

def growth(E, Om):
    la0 = np.log(1e-3)
    def rhs(la, y):
        a = np.exp(la); z = 1/a - 1
        h = 1e-5
        dlnE = (np.log(E(1/np.exp(la+h)-1)) - np.log(E(z))) / h
        return [y[1], -(2 + dlnE)*y[1] + 1.5*Om*(1+z)**3/E(z)**2 * y[0]]
    las = np.linspace(la0, 0, 4000)
    sol = solve_ivp(rhs, [la0, 0], [np.exp(la0), np.exp(la0)], t_eval=las, rtol=1e-10, atol=1e-14)
    D, Dp = sol.y
    f = Dp / D
    return las, D, f

Om_P, Or = 0.3153, 9.24e-5
Om_M = 1 - 0.6882 - Or
E_P = make_E_lcdm(Om_P, Or)
E_M = make_E_model(Om_M, Or)
laP, DP, fP = growth(E_P, Om_P)
laM, DM, fM = growth(E_M, Om_M)

s8_P = 0.8111                               # Planck 2018 TT,TE,EE+lowE+lensing
s8_M = s8_P * (DM[-1]/DM[0]) / (DP[-1]/DP[0])  # same early amplitude
print(f'sigma8 today:  LCDM {s8_P:.4f}   model {s8_M:.4f}   (suppression {s8_M/s8_P-1:+.2%})')
print(f'S8 = sigma8*sqrt(Om/0.3):  LCDM {s8_P*np.sqrt(Om_P/0.3):.4f}   model {s8_M*np.sqrt(Om_M/0.3):.4f}')

def fs8(la_arr, D, f, s8_0, z):
    la = np.log(1/(1+z))
    Dz = np.interp(la, la_arr, D)
    fz = np.interp(la, la_arr, f)
    return fz * s8_0 * Dz / D[-1]

# RSD compilation (classic, pre-2024, well-established values)
data = [
 ('6dFGS',      0.067, 0.423, 0.055),
 ('SDSS MGS',   0.15,  0.530, 0.160),
 ('BOSS DR12',  0.38,  0.497, 0.045),
 ('WiggleZ',    0.44,  0.413, 0.080),
 ('BOSS DR12',  0.51,  0.458, 0.038),
 ('WiggleZ',    0.60,  0.390, 0.063),
 ('BOSS DR12',  0.61,  0.436, 0.034),
 ('eBOSS LRG',  0.70,  0.473, 0.041),
 ('WiggleZ',    0.73,  0.437, 0.072),
 ('eBOSS ELG',  0.85,  0.315, 0.095),
 ('VIPERS',     0.86,  0.400, 0.110),
 ('FastSound',  1.40,  0.482, 0.116),
 ('eBOSS QSO',  1.48,  0.462, 0.045),
]
print('\n z     survey        obs +- err     LCDM    model   pull_L  pull_M')
chi2P = chi2M = 0
for name, z, obs, err in data:
    tP = fs8(laP, DP, fP, s8_P, z)
    tM = fs8(laM, DM, fM, s8_M, z)
    pP, pM = (obs-tP)/err, (obs-tM)/err
    chi2P += pP**2; chi2M += pM**2
    print(f'{z:5.3f}  {name:<12} {obs:.3f}+-{err:.3f}   {tP:.3f}   {tM:.3f}   {pP:+5.2f}  {pM:+5.2f}')
n = len(data)
print(f'\nchi2 (n={n}):  LCDM {chi2P:.2f}   model {chi2M:.2f}   Delta = {chi2M-chi2P:+.2f}')
print('(no parameters fit to this data in either case)')
