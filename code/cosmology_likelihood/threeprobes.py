# Full three-probe accounting: CMB distance priors + DESI DR2 BAO + SN shape.
# SN block: binned mu(z) generated from the compilations' published LCDM best fits
# (Pantheon+ Om=0.334+-0.018 or DES-SN5YR Om=0.352+-0.017), sigma scaled to match
# the published sigma(Om); free absolute magnitude M. Standard Gaussian compression.
import numpy as np
from scipy.optimize import minimize, minimize_scalar
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

ZS_SN = np.logspace(np.log10(0.025), np.log10(1.10), 30)

def lcdm_shape_mu(om):
    zz = np.linspace(0, 1.2, 3000)
    E = np.sqrt(om*(1+zz)**3 + (1-om))
    ch = np.concatenate(([0.0], cumulative_trapezoid(1.0/E, zz)))
    dm = np.interp(ZS_SN, zz, ch)
    return 5*np.log10((1+ZS_SN)*dm)

def calib_sn_sigma(om_c, sig_om):
    """Choose flat per-bin sigma so the shape fit yields sigma(Om) = sig_om."""
    mu0 = lcdm_shape_mu(om_c)
    d_om = 0.01
    dmu = lcdm_shape_mu(om_c + d_om) - mu0
    dmu -= dmu.mean()                      # M marginalization removes the mean
    # chi2(om) curvature: sum((dmu/d_om)^2/s^2)*(om-om_c)^2 = ((om-om_c)/sig_om)^2
    s = np.sqrt(np.sum((dmu/d_om)**2) ) * sig_om
    return s

def sn_chi2(mu_pred, mu_data, s):
    d = mu_data - mu_pred
    d -= d.mean()
    return np.sum((d/s)**2)

def make_bg(ombh2, omch2, h, model):
    p = camb.set_params(H0=100*h, ombh2=ombh2, omch2=omch2, mnu=MNU,
                        As=2.1e-9, ns=0.9656, tau=0.054)
    if model:
        om = (ombh2 + omch2 + MNU/93.14)/h**2
        orad = 2.4728e-5*(1+0.2271*3.044)/h**2
        z = np.concatenate(([0.0], np.logspace(-4, 4, 2500)))
        ode0 = 1 - om - orad
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
        for _ in range(8):
            w = -1 + C0/E
            I = np.concatenate(([0.0], cumulative_trapezoid(3*(1+w)/(1+z), z)))
            E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
        de = DarkEnergyPPF(); de.set_w_a_table(1/(1+z[::-1])[::-1]*0 + (1/(1+z))[::-1], (-1 + C0/E)[::-1])
        p.DarkEnergy = de
    return camb.get_results(p) if False else camb.get_background(p)

def total_chi2(params, model, mu_data, s_sn):
    if model:
        ombh2, h = params
        omch2 = 0.312*h*h - ombh2 - MNU/93.14
    else:
        ombh2, omch2, h = params
    if not (0.019 < ombh2 < 0.026 and 0.05 < omch2 < 0.25 and 0.58 < h < 0.78):
        return 1e10
    try:
        r = make_bg(ombh2, omch2, h, model)
    except Exception:
        return 1e10
    d = r.get_derived_params()
    om = (ombh2 + omch2 + MNU/93.14)/h**2
    DMs = r.comoving_radial_distance(d['zstar'])
    R = np.sqrt(om)*(100*h/C_KMS)*DMs
    lA = np.pi*DMs/d['rstar']
    v = np.array([R, lA, ombh2]) - PRI_MEAN
    x2 = v @ PRI_ICOV @ v
    zs = np.array([BAO_DV[0]] + [b[0] for b in BAO])
    DM = r.comoving_radial_distance(zs)
    DH = C_KMS/np.array([r.hubble_parameter(z) for z in zs])
    rd = d['rdrag']
    dv = (zs[0]*DM[0]**2*DH[0])**(1/3)/rd
    x2 += ((dv - BAO_DV[1])/BAO_DV[2])**2
    for i, (z, dmv, sm, dhv, sh, rho) in enumerate(BAO):
        dvec = np.array([DM[i+1]/rd - dmv, DH[i+1]/rd - dhv])
        cov = np.array([[sm*sm, rho*sm*sh], [rho*sm*sh, sh*sh]])
        x2 += dvec @ np.linalg.inv(cov) @ dvec
    mu_pred = 5*np.log10((1+ZS_SN)*r.comoving_radial_distance(ZS_SN))
    x2 += sn_chi2(mu_pred, mu_data, s_sn)
    return x2

for sn_name, om_sn, sig_om in (('Pantheon+ (Om=0.334+-0.018)', 0.334, 0.018),
                               ('DES-SN5YR original (Om=0.352+-0.017)', 0.352, 0.017),
                               ('DES-Dovekie 2026 (Om=0.330+-0.015)', 0.330, 0.015),
                               ('Union3 (Om=0.356+-0.027)', 0.356, 0.027)):
    s_sn = calib_sn_sigma(om_sn, sig_om)
    mu_data = lcdm_shape_mu(om_sn)
    r1 = minimize(total_chi2, [0.02253, 0.11816, 0.6854], args=(False, mu_data, s_sn),
                  method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-4, maxiter=3000))
    r2 = minimize(total_chi2, [0.02277, 0.6664], args=(True, mu_data, s_sn),
                  method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-4, maxiter=3000))
    om1 = (r1.x[0] + r1.x[1] + MNU/93.14)/r1.x[2]**2
    print(f'--- CMB + BAO + SN [{sn_name}] ---')
    print(f'   LCDM : chi2 = {r1.fun:7.2f}   Om = {om1:.4f}  H0 = {100*r1.x[2]:.2f}')
    print(f'   model: chi2 = {r2.fun:7.2f}   Om = 0.3120  H0 = {100*r2.x[1]:.2f}')
    print(f'   delta-chi2 (model - LCDM) = {r2.fun - r1.fun:+.2f}   (CMB+BAO only was +13.6)')
    print()
