import os
# Referee-grade tier: Planck 2018 native likelihoods (plik-lite TTTEEE + lowl TT + lowl EE)
# maximized for (a) LCDM and (b) the locked recursion model, with CAMB (exact w(z) table).
# Usage: python plikfit.py lcdm|model
import sys, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import minimize
import camb
from camb.dark_energy import DarkEnergyPPF
from cobaya.likelihoods.planck_2018_highl_plik.TTTEEE_lite_native import TTTEEE_lite_native
from cobaya.likelihoods.planck_2018_lowl.TT import TT as LowTT
from cobaya.likelihoods.planck_2018_lowl.EE import EE as LowEE

PK = os.environ.get('COBAYA_PACKAGES_PATH', 'cobaya_packages')
HI, LT, LE = TTTEEE_lite_native(packages_path=PK), LowTT(packages_path=PK), LowEE(packages_path=PK)
CASE = sys.argv[1]
C_DRAIN, MNU, NS_MODEL = 0.133, 0.06, 0.9656
LOG = open(rf'plikfit_{CASE}.log', 'a', encoding='utf-8')

def model_w_a(h, om):
    orad = 4.15e-5/h**2
    ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, 4, 3000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        I = np.concatenate(([0.0], np.cumsum(0.5*(3*(1+w[1:])/(1+z[1:]) + 3*(1+w[:-1])/(1+z[:-1]))*np.diff(z))))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    return (1/(1+z))[::-1], (-1 + C_DRAIN/E)[::-1]

def spectra(ombh2, omch2, H0, lnAs, ns, tau, model):
    p = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, mnu=MNU, As=np.exp(lnAs)*1e-10, ns=ns,
                        lmax=2700, lens_potential_accuracy=1)
    if model:
        h = H0/100
        om = (ombh2 + omch2 + MNU/93.14)/h**2
        a, w = model_w_a(h, om)
        de = DarkEnergyPPF(); de.set_w_a_table(a, w); p.DarkEnergy = de
    r = camb.get_results(p)
    cl = r.get_cmb_power_spectra(p, CMB_unit='muK')['total']   # D_l, mu K^2
    return cl, r, p

def neg2logL(cl, A_planck):
    x2_hi = HI.get_chi_squared(0, cl[:, 0], cl[:, 3], cl[:, 1], A_planck)
    l_tt = LT.log_likelihood(cl[:, 0], A_planck)
    l_ee = LE.log_likelihood(cl[:, 1], A_planck)
    if not np.isfinite(l_tt) or not np.isfinite(l_ee):
        return 1e9, (x2_hi, np.inf, np.inf)
    x2_prior = ((A_planck - 1.0)/0.0025)**2
    total = x2_hi - 2*l_tt - 2*l_ee + x2_prior
    return total, (x2_hi, -2*l_tt, -2*l_ee)

n_eval = [0]
best = [np.inf, None]
def objective(x):
    n_eval[0] += 1
    try:
        if CASE in ('lcdm', 'modelfree'):
            ombh2, omch2, H0, lnAs, ns, tau, Ap = x
            model = (CASE == 'modelfree')
        else:
            ombh2, H0, lnAs, tau, Ap = x
            ns = NS_MODEL
            h = H0/100
            omch2 = 0.312*h*h - ombh2 - MNU/93.14
            model = True
        if not (0.019 < ombh2 < 0.026 and 0.08 < omch2 < 0.16 and 60 < H0 < 76 and 2.9 < lnAs < 3.2
                and 0.93 < ns < 1.0 and 0.02 < tau < 0.12 and 0.98 < Ap < 1.02):
            return 1e9
        cl, r, p = spectra(ombh2, omch2, H0, lnAs, ns, tau, model)
        tot, parts = neg2logL(cl, Ap)
    except Exception as e:
        LOG.write(f'eval {n_eval[0]} failed: {e}\n'); LOG.flush()
        return 1e9
    if tot < best[0]:
        best[0], best[1] = tot, list(x)
    LOG.write(f'eval {n_eval[0]:4d}  -2lnL = {tot:9.3f}  [plik {parts[0]:8.3f}  lowTT {parts[1]:7.3f}  lowEE {parts[2]:7.3f}]  x = {np.array2string(np.array(x), precision=5)}\n')
    LOG.flush()
    return tot

if CASE in ('lcdm', 'modelfree'):
    x0 = np.array([0.02237, 0.1200, 67.36, 3.044, 0.9649, 0.0544, 1.0005])
    steps = np.array([0.00015, 0.0015, 0.5, 0.015, 0.004, 0.006, 0.002])
else:
    x0 = np.array([0.02260, 66.8, 3.044, 0.0544, 1.0005])
    steps = np.array([0.00015, 0.5, 0.015, 0.006, 0.002])
simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i]*steps[i] for i in range(len(x0))])
t0 = time.time()
LOG.write(f'\n=== start {CASE} {time.ctime()} ===\n'); LOG.flush()
res = minimize(objective, x0, method='Nelder-Mead',
               options=dict(initial_simplex=simplex, xatol=1e-4, fatol=0.02, maxfev=900, adaptive=True))
x = res.x
if CASE in ('lcdm', 'modelfree'):
    ombh2, omch2, H0, lnAs, ns, tau, Ap = x; model = (CASE == 'modelfree')
else:
    ombh2, H0, lnAs, tau, Ap = x; ns = NS_MODEL; h = H0/100; omch2 = 0.312*h*h - ombh2 - MNU/93.14; model = True
cl, r, p = spectra(ombh2, omch2, H0, lnAs, ns, tau, model)
tot, parts = neg2logL(cl, Ap)
d = r.get_derived_params()
h = H0/100; om = (ombh2 + omch2 + MNU/93.14)/h**2
p.set_matter_power(redshifts=[0.0], kmax=2.0); r2 = camb.get_results(p)
s8 = r2.get_sigma8_0()
out = dict(case=CASE, neg2logL=tot, plik=parts[0], lowTT=parts[1], lowEE=parts[2], ombh2=ombh2, omch2=omch2, H0=H0,
           lnAs=lnAs, ns=ns, tau=tau, A_planck=Ap, Om=om, thetastar=d['thetastar'], rdrag=d['rdrag'],
           sigma8=s8, S8=s8*np.sqrt(om/0.3), nfev=n_eval[0], minutes=(time.time()-t0)/60)
LOG.write('RESULT ' + json.dumps(out) + '\n'); LOG.flush()
print('RESULT', json.dumps(out, indent=1))
