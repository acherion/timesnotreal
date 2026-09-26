# Robustness: SN apparent-Omega_m under realistic weightings; BAO fit at alternate anchor.
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import cumulative_trapezoid

C_KMS = 299792.458
C_DRAIN = 0.133

def make_universe(h, om, model=True):
    orad = 4.15e-5/h**2
    ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, np.log10(3000), 6000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    if model:
        for _ in range(10):
            w = -1 + C_DRAIN/E
            I = np.concatenate(([0.0], cumulative_trapezoid(3*(1+w)/(1+z), z)))
            E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    return z, E

def lcdm_chi_tab(om, h=0.677):
    zz = np.linspace(0, 3.0, 6000)
    orad = 4.15e-5/h**2
    E = np.sqrt(om*(1+zz)**3 + orad*(1+zz)**4 + (1-om-orad))
    return zz, np.concatenate(([0.0], cumulative_trapezoid(1.0/E, zz)))

for (h_t, om_t) in ((0.677, 0.312), (0.681, 0.312)):
    zg, Eg = make_universe(h_t, om_t)
    chi_g = np.concatenate(([0.0], cumulative_trapezoid(1.0/Eg, zg)))
    def tDM(z): return (C_KMS/(100*h_t))*np.interp(z, zg, chi_g)

    print(f'truth anchor H0={100*h_t}, Om={om_t}:')
    # SN with three weightings
    for label, sig_f in (('flat 0.03', lambda z: np.full_like(z, 0.03)),
                         ('lowz-heavy', lambda z: 0.015 + 0.05*z),
                         ('highz-heavy', lambda z: 0.06 - 0.04*z/1.1)):
        zs = np.logspace(np.log10(0.025), np.log10(1.10), 40)
        mu_t = 5*np.log10((1+zs)*tDM(zs))
        sig = sig_f(zs)
        def chi2_om(om):
            zz, ch = lcdm_chi_tab(om, h_t)
            mu = 5*np.log10((1+zs)*(C_KMS/(100*h_t))*np.interp(zs, zz, ch))
            d = mu_t - mu
            M = np.sum(d/sig**2)/np.sum(1/sig**2)
            return np.sum(((d - M)/sig)**2)
        r = minimize_scalar(chi2_om, bounds=(0.15, 0.55), method='bounded')
        print(f'   SN apparent Omega_m ({label}): {r.x:.4f}')
print('real-world SN: Pantheon+ 0.334+-0.018, DES-SN5YR 0.352+-0.017, Union3 0.356+-0.027')
