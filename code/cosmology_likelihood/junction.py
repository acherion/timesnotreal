# Junction-window exploration: can a one-parameter modification of the w(z) tail
# rescue the CMB distance while preserving the pivot and BAO?
# Variant A (freeze): drain as committed below z_j; rho_DE frozen (w=-1) above z_j.
# Variant B (delayed drain, bookkeeping-consistent): drain begins at t_j = t(z_j);
#   rate rescaled so n reaches n0 today -> C -> C0 * t0/(t0 - t_j); w=-1 above z_j.
# Fit (ombh2, h, z_j) with Om = 0.312 locked against distance priors + DESI BAO.
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
import camb
from camb.dark_energy import DarkEnergyPPF

C_KMS = 299792.458
C0 = 0.133
MNU = 0.06

PRI_MEAN = np.array([1.7502, 301.471, 0.02236])
PRI_SIG  = np.array([0.0046, 0.090, 0.00015])
PRI_CORR = np.array([[1.0, 0.46, -0.66], [0.46, 1.0, -0.33], [-0.66, -0.33, 1.0]])
PRI_ICOV = np.linalg.inv(PRI_CORR * np.outer(PRI_SIG, PRI_SIG))

BAO_DV = (0.295, 7.944, 0.075)
BAO = [
    (0.510, 13.588, 0.167, 21.863, 0.427, -0.459),
    (0.706, 17.351, 0.177, 19.455, 0.330, -0.404),
    (0.934, 21.576, 0.152, 17.641, 0.193, -0.416),
    (1.321, 27.601, 0.318, 14.176, 0.221, -0.434),
    (1.484, 30.512, 0.760, 12.817, 0.516, -0.500),
    (2.330, 38.988, 0.531,  8.632, 0.101, -0.431),
]

def w_table(h, om, orad, zj, variant):
    z = np.concatenate(([0.0], np.logspace(-4, 4, 2500)))
    ode0 = 1 - om - orad
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    # effective drain constant
    for it in range(10):
        if variant == 'A':
            Ceff = C0
        else:
            # t(z_j)/t0 from current E(z): t(z) = int_z^inf dz'/((1+z')E'); t0 = t(0)
            integ = 1.0/((1+z)*E)
            tof = cumulative_trapezoid(integ, z, initial=0.0)
            t0 = tof[-1]
            tj = t0 - np.interp(zj, z, tof)
            frac = max(1 - tj/t0, 0.2)
            Ceff = C0/frac
        w = np.where(z <= zj, -1 + Ceff/E, -1.0)
        I = np.concatenate(([0.0], cumulative_trapezoid(3*(1+w)/(1+z), z)))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    a = 1/(1+z)
    w0 = w[0]
    wp = np.interp(0.34, z, w)
    return a[::-1], w[::-1], w0, wp

def chi2(p, variant):
    ombh2, h, zj = p
    if not (0.019 < ombh2 < 0.026 and 0.58 < h < 0.78 and 0.3 < zj < 50):
        return 1e10
    omch2 = 0.312*h*h - ombh2 - MNU/93.14
    try:
        pars = camb.set_params(H0=100*h, ombh2=ombh2, omch2=omch2, mnu=MNU,
                               As=2.1e-9, ns=0.9656, tau=0.054)
        om = 0.312
        orad = 2.4728e-5*(1 + 0.2271*3.044)/h**2
        a, w, w0, wp = w_table(h, om, orad, zj, variant)
        de = DarkEnergyPPF(); de.set_w_a_table(a, w)
        pars.DarkEnergy = de
        r = camb.get_background(pars)
        d = r.get_derived_params()
        DMs = r.comoving_radial_distance(d['zstar'])
        R = np.sqrt(om)*(100*h/C_KMS)*DMs
        lA = np.pi*DMs/d['rstar']
        v = np.array([R, lA, ombh2]) - PRI_MEAN
        x2 = v @ PRI_ICOV @ v
        zs = np.array([BAO_DV[0]] + [b[0] for b in BAO])
        DM = r.comoving_radial_distance(zs)
        Hz = np.array([r.hubble_parameter(z) for z in zs])
        DH = C_KMS/Hz
        rd = d['rdrag']
        dv = (zs[0]*DM[0]**2*DH[0])**(1/3)/rd
        x2 += ((dv - BAO_DV[1])/BAO_DV[2])**2
        for i, (z, dmv, sm, dhv, sh, rho) in enumerate(BAO):
            dvec = np.array([DM[i+1]/rd - dmv, DH[i+1]/rd - dhv])
            cov = np.array([[sm*sm, rho*sm*sh], [rho*sm*sh, sh*sh]])
            x2 += dvec @ np.linalg.inv(cov) @ dvec
        return x2
    except Exception:
        return 1e10

for variant, label in (('A', 'freeze above z_j (C = 0.133 below)'),
                       ('B', 'delayed drain, rate rescaled for n0-today')):
    best = None
    for zj0 in (1.0, 1.7, 3.0, 6.0):
        r = minimize(chi2, [0.02250, 0.680, zj0], args=(variant,), method='Nelder-Mead',
                     options=dict(xatol=1e-5, fatol=1e-4, maxiter=3000))
        if best is None or r.fun < best.fun: best = r
    ombh2, h, zj = best.x
    omch2 = 0.312*h*h - ombh2 - MNU/93.14
    om = 0.312
    orad = 2.4728e-5*(1+0.2271*3.044)/h**2
    a, w, w0, wp = w_table(h, om, orad, zj, variant)
    print(f'variant {variant} ({label}):')
    print(f'   chi2 = {best.fun:.2f}   z_j = {zj:.2f}   H0 = {100*h:.2f}   ombh2 = {ombh2:.5f}')
    print(f'   w0 = {w0:.4f}   w(0.34) = {wp:.4f}   [committed: w0 = -0.867, pivot -0.892; measured pivot -0.9 +- 0.1]')
    print()
print('reference: LCDM chi2 = 14.65; locked model chi2 = 28.30 on the same data')
