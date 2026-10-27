import os, sys, time, json
# SPT-3G D1 TT/TE/EE (SPTlite, candl) cross-check of the acoustic-scale squeeze: LCDM vs the locked recursion model,
# SPT alone (with candl's default tau and Tcal priors) or jointly with DESI DR2 BAO + SN shape.
# Usage: python sptfit.py <spt|joint> <lcdm|model>
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
import camb
from camb.dark_energy import DarkEnergyPPF
import candl, spt_candl_data
LIKE = candl.Like(spt_candl_data.SPT3G_D1_TnE_lite)
ELL = np.arange(LIKE.ell_min, LIKE.ell_max + 1)
TAU0, TAU_SIG, TCAL_SIG = 0.051, 0.006, 0.0036          # candl's default Gaussian priors for this likelihood
DATA, CASE = sys.argv[1], sys.argv[2]
C_DRAIN, MNU, NS_MODEL, C_KMS = 0.133, 0.06, 0.9656, 299792.458
LOG = open(rf'sptfit_{DATA}_{CASE}.log', 'a', encoding='utf-8')
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

def evaluate(ombh2, omch2, H0, lnAs, ns, tau, Tcal, Ecal, model):
    p = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, mnu=MNU, As=np.exp(lnAs)*1e-10, ns=ns, lmax=4600, lens_potential_accuracy=1)
    if model:
        h = H0/100; om = (ombh2 + omch2 + MNU/93.14)/h**2
        a, w = model_w_a(h, om); de = DarkEnergyPPF(); de.set_w_a_table(a, w); p.DarkEnergy = de
    r = camb.get_results(p)
    cl = r.get_cmb_power_spectra(p, CMB_unit='muK')['total']
    Dl = {'TT': cl[ELL, 0], 'TE': cl[ELL, 3], 'EE': cl[ELL, 1], 'ell': ELL}
    x_spt = float(LIKE.chi_square(dict(Dl=Dl, Tcal=Tcal, Ecal=Ecal, tau=tau)))
    x_cmb = x_spt + ((tau - TAU0)/TAU_SIG)**2 + ((Tcal - 1.0)/TCAL_SIG)**2
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
    return x_cmb + x_bao + x_sn, (x_cmb, x_spt, x_bao, x_sn, d['thetastar'], rd)

n_eval = [0]; best = [np.inf, None, None]
def objective(x):
    n_eval[0] += 1
    try:
        if CASE == 'lcdm':
            ombh2, omch2, H0, lnAs, ns, tau, Tcal, Ecal = x; model = False
        else:
            ombh2, H0, lnAs, tau, Tcal, Ecal = x; ns = NS_MODEL; h = H0/100; omch2 = 0.312*h*h - ombh2 - MNU/93.14; model = True
        if not (0.019 < ombh2 < 0.026 and 0.08 < omch2 < 0.16 and 58 < H0 < 78 and 2.9 < lnAs < 3.2 and 0.92 < ns < 1.0 and 0.02 < tau < 0.12 and 0.97 < Tcal < 1.03 and 0.95 < Ecal < 1.05):
            return 1e9
        tot, parts = evaluate(ombh2, omch2, H0, lnAs, ns, tau, Tcal, Ecal, model)
    except Exception as e:
        LOG.write(f'eval {n_eval[0]} failed: {e!r}\n'); LOG.flush(); return 1e9
    if not np.isfinite(tot): return 1e9
    if tot < best[0]: best[0], best[1], best[2] = tot, list(x), parts
    LOG.write(f'eval {n_eval[0]:4d}  total = {tot:9.3f}  [CMB {parts[0]:8.3f} (SPT {parts[1]:8.3f}) BAO {parts[2]:6.2f} SN {parts[3]:5.2f}]  x = {np.array2string(np.array(x), precision=5)}\n'); LOG.flush()
    return tot

if CASE == 'lcdm':
    x0 = np.array([0.02237, 0.1200, 67.0, 3.04, 0.960, 0.051, 1.0, 1.0]); steps = np.array([0.0002, 0.002, 0.8, 0.02, 0.006, 0.005, 0.003, 0.006])
else:
    x0 = np.array([0.02265, 66.6, 3.04, 0.051, 1.0, 1.0]); steps = np.array([0.0002, 0.8, 0.02, 0.005, 0.003, 0.006])
simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i]*steps[i] for i in range(len(x0))])
t0 = time.time(); LOG.write(f'\n=== start {DATA} {CASE} {time.ctime()} ===\n'); LOG.flush()
res = minimize(objective, x0, method='Nelder-Mead', options=dict(initial_simplex=simplex, xatol=1e-4, fatol=0.02, maxfev=1200, adaptive=True))
x = best[1]; parts = best[2]
if CASE == 'lcdm':
    ombh2, omch2, H0, lnAs, ns, tau, Tcal, Ecal = x
else:
    ombh2, H0, lnAs, tau, Tcal, Ecal = x; ns = NS_MODEL; h = H0/100; omch2 = 0.312*h*h - ombh2 - MNU/93.14
h = H0/100; om = (ombh2 + omch2 + MNU/93.14)/h**2
out = dict(data=DATA, case=CASE, total=best[0], CMB=parts[0], SPT=parts[1], BAO=parts[2], SN=parts[3], ombh2=ombh2, omch2=omch2, H0=H0, lnAs=lnAs, ns=ns, tau=tau, Tcal=Tcal, Ecal=Ecal, Om=om, thetastar=parts[4], rdrag=parts[5], nfev=n_eval[0], minutes=(time.time()-t0)/60)
LOG.write('RESULT ' + json.dumps(out) + '\n'); LOG.flush(); print('RESULT', json.dumps(out, indent=1))
