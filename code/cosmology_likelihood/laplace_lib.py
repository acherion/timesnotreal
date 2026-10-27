import os, sys, time, json
# Laplace-approximation evidence for LCDM vs the locked model on the joint likelihood
# (Planck native plik-lite TTTEEE + lowl TT + lowl EE + DESI DR2 BAO + SN shape), around the published best fits.
# ln Z ~ ln L_max + (k/2) ln 2pi + 0.5 ln|Sigma| - ln V_prior   (flat priors over the fit bounds; A_planck's Gaussian prior is inside chi2 for both)
# Usage: python laplace.py lcdm|model
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.integrate import cumulative_trapezoid
import camb
from camb.dark_energy import DarkEnergyPPF
from cobaya.likelihoods.planck_2018_highl_plik.TTTEEE_lite_native import TTTEEE_lite_native
from cobaya.likelihoods.planck_2018_lowl.TT import TT as LowTT
from cobaya.likelihoods.planck_2018_lowl.EE import EE as LowEE
PK = os.environ.get('COBAYA_PACKAGES_PATH', 'cobaya_packages')
HI, LT, LE = TTTEEE_lite_native(packages_path=PK), LowTT(packages_path=PK), LowEE(packages_path=PK)
CASE = 'lcdm'  # set by caller
C_DRAIN, MNU, NS_MODEL, C_KMS = 0.133, 0.06, 0.9656, 299792.458
BAO_DV = (0.295, 7.944, 0.075)
BAO = [(0.510, 13.588, 0.167, 21.863, 0.427, -0.459), (0.706, 17.351, 0.177, 19.455, 0.330, -0.404),
       (0.934, 21.576, 0.152, 17.641, 0.193, -0.416), (1.321, 27.601, 0.318, 14.176, 0.221, -0.434),
       (1.484, 30.512, 0.760, 12.817, 0.516, -0.500), (2.330, 38.988, 0.531,  8.632, 0.101, -0.431)]
ZS_SN = np.logspace(np.log10(0.025), np.log10(1.10), 30)
def sn_shape_mu(om):
    zz = np.linspace(0, 1.2, 3000); E = np.sqrt(om*(1+zz)**3 + (1-om))
    ch = np.concatenate(([0.0], cumulative_trapezoid(1.0/E, zz)))
    return 5*np.log10((1+ZS_SN)*np.interp(ZS_SN, zz, ch))
MU_SN = sn_shape_mu(0.330); _d = sn_shape_mu(0.340) - MU_SN; _d -= _d.mean(); S_SN = np.sqrt(np.sum((_d/0.01)**2))*0.015
def model_w_a(h, om):
    orad = 4.15e-5/h**2; ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, 4, 3000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        I = np.concatenate(([0.0], np.cumsum(0.5*(3*(1+w[1:])/(1+z[1:]) + 3*(1+w[:-1])/(1+z[:-1]))*np.diff(z))))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    return (1/(1+z))[::-1], (-1 + C_DRAIN/E)[::-1]
def chi2(x):
    if CASE == 'lcdm': ombh2, omch2, H0, lnAs, ns, tau, Ap = x; model = False
    else: ombh2, H0, lnAs, tau, Ap = x; ns = NS_MODEL; h = H0/100; omch2 = 0.312*h*h - ombh2 - MNU/93.14; model = True
    p = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, mnu=MNU, As=np.exp(lnAs)*1e-10, ns=ns, lmax=2700, lens_potential_accuracy=1)
    if model:
        h = H0/100; om = (ombh2 + omch2 + MNU/93.14)/h**2
        a, w = model_w_a(h, om); de = DarkEnergyPPF(); de.set_w_a_table(a, w); p.DarkEnergy = de
    r = camb.get_results(p); cl = r.get_cmb_power_spectra(p, CMB_unit='muK')['total']
    x_hi = HI.get_chi_squared(0, cl[:, 0], cl[:, 3], cl[:, 1], Ap)
    l_tt = LT.log_likelihood(cl[:, 0], Ap); l_ee = LE.log_likelihood(cl[:, 1], Ap)
    x_cmb = x_hi - 2*l_tt - 2*l_ee + ((Ap - 1.0)/0.0025)**2
    d = r.get_derived_params(); rd = d['rdrag']
    zs = np.array([BAO_DV[0]] + [b[0] for b in BAO])
    DM = r.comoving_radial_distance(zs); DH = C_KMS/np.array([r.hubble_parameter(z) for z in zs])
    x_bao = (((zs[0]*DM[0]**2*DH[0])**(1/3)/rd - BAO_DV[1])/BAO_DV[2])**2
    for i, (z, dmv, sm, dhv, sh, rho) in enumerate(BAO):
        dvec = np.array([DM[i+1]/rd - dmv, DH[i+1]/rd - dhv]); cov = np.array([[sm*sm, rho*sm*sh], [rho*sm*sh, sh*sh]])
        x_bao += dvec @ np.linalg.inv(cov) @ dvec
    mu = 5*np.log10((1+ZS_SN)*r.comoving_radial_distance(ZS_SN)); dd = MU_SN - mu; dd -= dd.mean()
    return float(x_cmb + x_bao + np.sum((dd/S_SN)**2))

