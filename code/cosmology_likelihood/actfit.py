import os
# Independent CMB check: ACT DR6 CMB-only (l=600-6500 TT/TE/EE) + Planck 2018 low-ell EE (tau).
# Usage: python actfit.py lcdm|model
import sys, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import minimize
import camb
from camb.dark_energy import DarkEnergyPPF
from act_dr6_cmbonly import ACTDR6CMBonly
from cobaya.likelihoods.planck_2018_lowl.EE import EE as LowEE
PK = os.environ.get('COBAYA_PACKAGES_PATH', 'cobaya_packages')
ACT, LE = ACTDR6CMBonly(packages_path=PK), LowEE(packages_path=PK)
CASE = sys.argv[1]
C_DRAIN, MNU, NS_MODEL = 0.133, 0.06, 0.9656
LOG = open(rf'actfit_{CASE}.log', 'a', encoding='utf-8')

def model_w_a(h, om):
    orad = 4.15e-5/h**2; ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, 4, 3000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        I = np.concatenate(([0.0], np.cumsum(0.5*(3*(1+w[1:])/(1+z[1:]) + 3*(1+w[:-1])/(1+z[:-1]))*np.diff(z))))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    return (1/(1+z))[::-1], (-1 + C_DRAIN/E)[::-1]

def spectra(ombh2, omch2, H0, lnAs, ns, tau, model):
    p = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, mnu=MNU, As=np.exp(lnAs)*1e-10, ns=ns,
                        lmax=7000, lens_potential_accuracy=1)
    if model:
        h = H0/100; om = (ombh2 + omch2 + MNU/93.14)/h**2
        a, w = model_w_a(h, om); de = DarkEnergyPPF(); de.set_w_a_table(a, w); p.DarkEnergy = de
    r = camb.get_results(p)
    cl = r.get_cmb_power_spectra(p, CMB_unit='muK')['total']
    pad = np.zeros((9001, 4)); n = min(len(cl), 9001); pad[:n] = cl[:n]
    return pad, r, p

def neg2logL(cl, A_act, P_act):
    x2_act = ACT.chi_square({'tt': cl[:, 0], 'te': cl[:, 3], 'ee': cl[:, 1]}, A_act, P_act)
    l_ee = LE.log_likelihood(cl[:, 1], 1.0)
    if not np.isfinite(l_ee): return 1e9, (x2_act, np.inf)
    prior = ((A_act - 1.0)/0.01)**2 + ((P_act - 1.0)/0.02)**2      # calibration / pol-efficiency priors (~1% / ~2%)
    return x2_act - 2*l_ee + prior, (x2_act, -2*l_ee)

n_eval = [0]
def objective(x):
    n_eval[0] += 1
    try:
        if CASE == 'lcdm':
            ombh2, omch2, H0, lnAs, ns, tau, Aa, Pa = x; model = False
        else:
            ombh2, H0, lnAs, tau, Aa, Pa = x; ns = NS_MODEL; h = H0/100
            omch2 = 0.312*h*h - ombh2 - MNU/93.14; model = True
        if not (0.019 < ombh2 < 0.026 and 0.08 < omch2 < 0.16 and 60 < H0 < 76 and 2.9 < lnAs < 3.2
                and 0.93 < ns < 1.0 and 0.02 < tau < 0.12 and 0.9 < Aa < 1.1 and 0.9 < Pa < 1.1):
            return 1e9
        cl, r, p = spectra(ombh2, omch2, H0, lnAs, ns, tau, model)
        tot, parts = neg2logL(cl, Aa, Pa)
    except Exception as e:
        import traceback
        LOG.write(f'eval {n_eval[0]} failed: {e!r}\n' + traceback.format_exc() + '\n'); LOG.flush(); return 1e9
    LOG.write(f'eval {n_eval[0]:4d}  -2lnL = {tot:9.3f}  [ACT {parts[0]:8.3f}  lowEE {parts[1]:7.3f}]  x = {np.array2string(np.array(x), precision=5)}\n'); LOG.flush()
    return tot

if CASE == 'lcdm':
    x0 = np.array([0.02237, 0.1200, 67.36, 3.044, 0.9649, 0.0544, 1.0, 1.0])
    steps = np.array([0.0002, 0.002, 0.7, 0.02, 0.005, 0.006, 0.005, 0.01])
else:
    x0 = np.array([0.02260, 66.6, 3.04, 0.0544, 1.0, 1.0])
    steps = np.array([0.0002, 0.7, 0.02, 0.006, 0.005, 0.01])
simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i]*steps[i] for i in range(len(x0))])
t0 = time.time(); LOG.write(f'\n=== start {CASE} {time.ctime()} ===\n'); LOG.flush()
res = minimize(objective, x0, method='Nelder-Mead', options=dict(initial_simplex=simplex, xatol=1e-4, fatol=0.02, maxfev=1000, adaptive=True))
x = res.x
if CASE == 'lcdm':
    ombh2, omch2, H0, lnAs, ns, tau, Aa, Pa = x; model = False
else:
    ombh2, H0, lnAs, tau, Aa, Pa = x; ns = NS_MODEL; h = H0/100; omch2 = 0.312*h*h - ombh2 - MNU/93.14; model = True
cl, r, p = spectra(ombh2, omch2, H0, lnAs, ns, tau, model)
tot, parts = neg2logL(cl, Aa, Pa)
d = r.get_derived_params(); h = H0/100; om = (ombh2 + omch2 + MNU/93.14)/h**2
p.set_matter_power(redshifts=[0.0], kmax=2.0); r2 = camb.get_results(p); s8 = r2.get_sigma8_0()
out = dict(case=CASE, neg2logL=tot, act=parts[0], lowEE=parts[1], ombh2=ombh2, omch2=omch2, H0=H0, lnAs=lnAs, ns=ns, tau=tau,
           A_act=Aa, P_act=Pa, Om=om, thetastar=d['thetastar'], rdrag=d['rdrag'], sigma8=s8, S8=s8*np.sqrt(om/0.3), nfev=n_eval[0], minutes=(time.time()-t0)/60)
LOG.write('RESULT ' + json.dumps(out) + '\n'); LOG.flush(); print('RESULT', json.dumps(out, indent=1))
