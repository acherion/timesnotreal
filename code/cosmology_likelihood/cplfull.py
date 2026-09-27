import os, sys, time, json
# Benchmark: LCDM + CPL (w0, wa) with all standard parameters free, against the same full likelihood
# (Planck native plik-lite TTTEEE + lowl TT + lowl EE [+ DESI DR2 BAO + SN shape]).
# Usage: python cplfull.py <planck|joint>
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import minimize
import camb
from camb.dark_energy import DarkEnergyPPF
from cobaya.likelihoods.planck_2018_highl_plik.TTTEEE_lite_native import TTTEEE_lite_native
from cobaya.likelihoods.planck_2018_lowl.TT import TT as LowTT
from cobaya.likelihoods.planck_2018_lowl.EE import EE as LowEE
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inflow_bg import BAO, BAO_DV, ZS_SN, MU_SN, S_SN, MNU, C_KMS
PK = os.environ.get('COBAYA_PACKAGES_PATH', 'cobaya_packages')
HI, LT, LE = TTTEEE_lite_native(packages_path=PK), LowTT(packages_path=PK), LowEE(packages_path=PK)
DATA = sys.argv[1]
LOG = open(f'cplfull_{DATA}.log', 'a', encoding='utf-8')

def evaluate(ombh2, omch2, H0, lnAs, ns, tau, Ap, w0, wa):
    p = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, mnu=MNU, As=np.exp(lnAs)*1e-10, ns=ns, lmax=2700, lens_potential_accuracy=1)
    de = DarkEnergyPPF(); de.set_params(w=w0, wa=wa); p.DarkEnergy = de
    r = camb.get_results(p)
    cl = r.get_cmb_power_spectra(p, CMB_unit='muK')['total']
    x_hi = HI.get_chi_squared(0, cl[:, 0], cl[:, 3], cl[:, 1], Ap)
    l_tt = LT.log_likelihood(cl[:, 0], Ap); l_ee = LE.log_likelihood(cl[:, 1], Ap)
    if not (np.isfinite(l_tt) and np.isfinite(l_ee)): return 1e9, None
    x_cmb = x_hi - 2*l_tt - 2*l_ee + ((Ap - 1.0)/0.0025)**2
    d = r.get_derived_params(); rd = d['rdrag']
    x_bao = x_sn = 0.0
    if DATA == 'joint':
        zs = np.array([BAO_DV[0]] + [b[0] for b in BAO])
        DM = r.comoving_radial_distance(zs); DH = C_KMS/np.array([r.hubble_parameter(z) for z in zs])
        x_bao = (((zs[0]*DM[0]**2*DH[0])**(1/3)/rd - BAO_DV[1])/BAO_DV[2])**2
        for i, (z, dmv, sm, dhv, sh, rho) in enumerate(BAO):
            dvec = np.array([DM[i+1]/rd - dmv, DH[i+1]/rd - dhv]); cov = np.array([[sm*sm, rho*sm*sh], [rho*sm*sh, sh*sh]])
            x_bao += dvec @ np.linalg.inv(cov) @ dvec
        mu = 5*np.log10((1+ZS_SN)*r.comoving_radial_distance(ZS_SN)); dd = MU_SN - mu; dd -= dd.mean()
        x_sn = float(np.sum((dd/S_SN)**2))
    return x_cmb + x_bao + x_sn, dict(CMB=x_cmb, BAO=x_bao, SN=x_sn, thetastar=d['thetastar'], rdrag=rd)

n_eval = [0]; best = [np.inf, None, None]
def objective(x):
    n_eval[0] += 1
    ombh2, omch2, H0, lnAs, ns, tau, Ap, w0, wa = x
    if not (0.019 < ombh2 < 0.026 and 0.08 < omch2 < 0.16 and 55 < H0 < 80 and 2.9 < lnAs < 3.2 and 0.93 < ns < 1.0
            and 0.02 < tau < 0.12 and 0.98 < Ap < 1.02 and -2.0 < w0 < -0.3 and -3.5 < wa < 1.5 and w0 + wa < 0.2):
        return 1e9
    try:
        tot, info = evaluate(*x)
    except Exception as e:
        LOG.write(f'eval {n_eval[0]} failed: {e}\n'); LOG.flush(); return 1e9
    if info is None: return 1e9
    if tot < best[0]: best[0], best[1], best[2] = tot, list(x), info
    LOG.write(f"eval {n_eval[0]:4d}  total = {tot:9.3f}  [CMB {info['CMB']:8.3f} BAO {info['BAO']:6.2f} SN {info['SN']:5.2f}]  x = {np.array2string(np.array(x), precision=5)}\n"); LOG.flush()
    return tot

x0 = np.array([0.02237, 0.1190, 66.5, 3.044, 0.9660, 0.0544, 1.0005, -0.80, -0.70])
steps = np.array([0.00015, 0.0015, 0.8, 0.015, 0.004, 0.006, 0.002, 0.08, 0.3])
simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i]*steps[i] for i in range(len(x0))])
t0 = time.time(); LOG.write(f'\n=== start {DATA} {time.ctime()} ===\n'); LOG.flush()
res = minimize(objective, x0, method='Nelder-Mead', options=dict(initial_simplex=simplex, xatol=1e-4, fatol=0.02, maxfev=1800, adaptive=True))
x = best[1]; info = best[2]
ombh2, omch2, H0, lnAs, ns, tau, Ap, w0, wa = x
h = H0/100; om = (ombh2 + omch2 + MNU/93.14)/h**2
out = dict(data=DATA, total=best[0], ombh2=ombh2, omch2=omch2, H0=H0, lnAs=lnAs, ns=ns, tau=tau, A_planck=Ap, w0=w0, wa=wa, Om=om,
           nfev=n_eval[0], minutes=(time.time()-t0)/60, **info)
LOG.write('RESULT ' + json.dumps(out, default=float) + '\n'); LOG.flush()
print('RESULT', json.dumps(out, indent=1, default=float))
