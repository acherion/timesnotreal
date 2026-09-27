import os, sys, time, json
# Full-likelihood tier for the inflow door: Planck 2018 native (plik-lite TTTEEE + lowl TT + lowl EE),
# optionally + DESI DR2 BAO (13 pts) + SN shape (0.330+-0.015), for the locked recursion model
# (Omega_m = 0.312, n_s = 0.9656, committed drain) plus a parent-inflow source in the DE continuity equation.
# Usage: python inflowfit.py <planck|joint> <fixed|free> <family> [params...]
#   fixed: inflow params fixed at the given values; free: inflow params profiled (start at given values)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import minimize
import camb
from camb.dark_energy import DarkEnergyPPF
from cobaya.likelihoods.planck_2018_highl_plik.TTTEEE_lite_native import TTTEEE_lite_native
from cobaya.likelihoods.planck_2018_lowl.TT import TT as LowTT
from cobaya.likelihoods.planck_2018_lowl.EE import EE as LowEE
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inflow_bg import de_history, BAO, BAO_DV, ZS_SN, MU_SN, S_SN, MNU, NS_MODEL, C_KMS, OM_LOCK

PK = os.environ.get('COBAYA_PACKAGES_PATH', 'cobaya_packages')
HI, LT, LE = TTTEEE_lite_native(packages_path=PK), LowTT(packages_path=PK), LowEE(packages_path=PK)
DATA, CASE, KIND = sys.argv[1], sys.argv[2], sys.argv[3]
PRM0 = [float(v) for v in sys.argv[4:]]
TAG = f'{DATA}_{CASE}_{KIND}_' + '_'.join(f'{v:g}' for v in PRM0)
LOG = open(f'inflowfit_{TAG}.log', 'a', encoding='utf-8')
NP = len(PRM0)

def evaluate(ombh2, H0, lnAs, tau, Ap, prm):
    h = H0/100
    omch2 = OM_LOCK*h*h - ombh2 - MNU/93.14
    hist = de_history(h, OM_LOCK, KIND, tuple(prm))
    if hist is None: return 1e9, None
    p = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, mnu=MNU, As=np.exp(lnAs)*1e-10, ns=NS_MODEL,
                        lmax=2700, lens_potential_accuracy=1)
    de = DarkEnergyPPF(); de.set_w_a_table(hist['a'], hist['w']); p.DarkEnergy = de
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
    tot = x_cmb + x_bao + x_sn
    info = dict(CMB=x_cmb, plik=x_hi, lowTT=-2*l_tt, lowEE=-2*l_ee, BAO=x_bao, SN=x_sn, thetastar=d['thetastar'], rdrag=rd,
                F=hist['F'], fM=hist['fM'], z_cross=hist['z_cross'], w_min=hist['w_min'], w0=hist['w0'], t0=hist['t0'])
    return tot, info

def in_bounds(ombh2, H0, lnAs, tau, Ap, prm):
    if not (0.019 < ombh2 < 0.026 and 58 < H0 < 78 and 2.9 < lnAs < 3.2 and 0.02 < tau < 0.12 and 0.98 < Ap < 1.02): return False
    if KIND == 'tophat': return 0 < prm[0] < 6 and 0.02 < prm[1] < 6
    if KIND == 'expdec': return 0 < prm[0] < 8 and 0.3 < prm[1] < 40
    if KIND == 'gauss':  return 0 < prm[0] < 8 and 0.1 < prm[1] < 6 and 0.08 < prm[2] < 2.0
    return True

n_eval = [0]; best = [np.inf, None, None]
def objective(x):
    n_eval[0] += 1
    ombh2, H0, lnAs, tau, Ap = x[:5]
    prm = list(x[5:]) if CASE == 'free' else ([x[5]] + PRM0[1:] if CASE == 'freeA' else PRM0)
    if not in_bounds(ombh2, H0, lnAs, tau, Ap, prm): return 1e9
    try:
        tot, info = evaluate(ombh2, H0, lnAs, tau, Ap, prm)
    except Exception as e:
        LOG.write(f'eval {n_eval[0]} failed: {e}\n'); LOG.flush(); return 1e9
    if info is None: return 1e9
    if tot < best[0]: best[0], best[1], best[2] = tot, list(x), info
    LOG.write(f"eval {n_eval[0]:4d}  total = {tot:9.3f}  [CMB {info['CMB']:8.3f} BAO {info['BAO']:6.2f} SN {info['SN']:5.2f}]  "
              f"F={info['F']:.3f} zx={[round(v,2) for v in info['z_cross']]}  x = {np.array2string(np.array(x), precision=5)}\n"); LOG.flush()
    return tot

x0 = [0.02265, 66.8, 3.04, 0.0544, 1.0005]; steps = [0.00015, 0.6, 0.015, 0.006, 0.002]
if CASE == 'free':
    x0 += PRM0
    steps += {'tophat': [0.15, 0.15], 'expdec': [0.15, 1.0], 'gauss': [0.15, 0.15, 0.08]}[KIND]
elif CASE == 'freeA':
    x0 += [PRM0[0]]; steps += [0.15]
x0 = np.array(x0); steps = np.array(steps)
simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i]*steps[i] for i in range(len(x0))])
t0 = time.time(); LOG.write(f'\n=== start {TAG} {time.ctime()} ===\n'); LOG.flush()
res = minimize(objective, x0, method='Nelder-Mead', options=dict(initial_simplex=simplex, xatol=1e-4, fatol=0.02, maxfev=1400, adaptive=True))
x = best[1]; info = best[2]
ombh2, H0, lnAs, tau, Ap = x[:5]; prm = list(x[5:]) if CASE == 'free' else ([x[5]] + PRM0[1:] if CASE == 'freeA' else PRM0)
h = H0/100; omch2 = OM_LOCK*h*h - ombh2 - MNU/93.14
out = dict(tag=TAG, data=DATA, case=CASE, kind=KIND, prm=prm, total=best[0], ombh2=ombh2, omch2=omch2, H0=H0, lnAs=lnAs, tau=tau, A_planck=Ap,
           nfev=n_eval[0], minutes=(time.time()-t0)/60, **info)
LOG.write('RESULT ' + json.dumps(out, default=float) + '\n'); LOG.flush()
print('RESULT', json.dumps(out, indent=1, default=float))
