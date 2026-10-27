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
CASE = sys.argv[1]
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

if CASE == 'lcdm':
    x0 = np.array([0.022516, 0.117772, 68.330, 3.04265, 0.97047, 0.055975, 1.000785])
    h = np.array([1.5e-4, 1.2e-3, 0.5, 0.015, 0.004, 0.006, 0.002])
    names = ['ombh2', 'omch2', 'H0', 'lnAs', 'ns', 'tau', 'A_planck']
    widths = np.array([0.007, 0.08, 16.0, 0.3, 0.07, 0.10, None], dtype=object)
else:
    x0 = np.array([0.022769, 66.509, 3.03805, 0.056291, 1.000793])
    h = np.array([1.5e-4, 0.5, 0.015, 0.006, 0.002])
    names = ['ombh2', 'H0', 'lnAs', 'tau', 'A_planck']
    widths = np.array([0.007, 16.0, 0.3, 0.10, None], dtype=object)
n = len(x0); t0 = time.time()
f0 = chi2(x0); print(f'{CASE}: chi2(x0) = {f0:.3f}')
g = np.zeros(n); Hm = np.zeros((n, n)); fp = np.zeros(n); fm = np.zeros(n)
for i in range(n):
    e = np.zeros(n); e[i] = h[i]
    fp[i] = chi2(x0 + e); fm[i] = chi2(x0 - e)
    g[i] = (fp[i] - fm[i])/(2*h[i]); Hm[i, i] = (fp[i] - 2*f0 + fm[i])/h[i]**2
    print(f'  diag {names[i]}: f+={fp[i]:.3f} f-={fm[i]:.3f} grad={g[i]:.4g} H_ii={Hm[i,i]:.4g}', flush=True)
for i in range(n):
    for j in range(i+1, n):
        ei = np.zeros(n); ei[i] = h[i]; ej = np.zeros(n); ej[j] = h[j]
        fpp = chi2(x0 + ei + ej); fmm = chi2(x0 - ei - ej)
        Hm[i, j] = Hm[j, i] = (fpp - fp[i] - fp[j] + 2*f0 - fm[i] - fm[j] + fmm)/(2*h[i]*h[j])
        print(f'  offdiag {names[i]},{names[j]}: H={Hm[i,j]:.4g}', flush=True)
# Newton refinement of the minimum and posterior covariance (chi2 = -2 lnL => Sigma = 2 H^-1)
w, V = np.linalg.eigh(Hm); print('Hessian eigenvalues:', w)
Hpos = V @ np.diag(np.maximum(w, 1e-12)) @ V.T
Hinv = np.linalg.inv(Hpos)
dx = -Hinv @ g; chi2_min = f0 - 0.5*g @ Hinv @ g
Sigma = 2*Hinv
sig = np.sqrt(np.diag(Sigma))
print('Newton step:', dict(zip(names, dx))); print('chi2_min (quadratic):', chi2_min)
print('posterior sigmas:', dict(zip(names, sig)))
# evidence pieces: flat priors over stated widths except A_planck (Gaussian prior, inside chi2, shared by both models)
k = n
sign, logdet = np.linalg.slogdet(Sigma)
lnV = float(np.sum([np.log(wd) for wd in widths if wd is not None]))
lnZ = -0.5*chi2_min + 0.5*k*np.log(2*np.pi) + 0.5*logdet - lnV
out = dict(case=CASE, chi2_x0=f0, chi2_min_quadratic=chi2_min, k=k, logdet_Sigma=logdet, lnV_prior=lnV, lnZ_relative=lnZ,
           occam=0.5*k*np.log(2*np.pi) + 0.5*logdet - lnV, sigmas=dict(zip(names, sig.tolist())), newton_step=dict(zip(names, dx.tolist())), minutes=(time.time()-t0)/60)
print('RESULT', json.dumps(out, indent=1))
open(f'laplace_{CASE}.json', 'w').write(json.dumps(out, indent=1))
