import os, sys, json, glob
# Post-process inflow full-likelihood fits: growth (sigma8, S8, fsigma8), effective w(z), birth reservoir, CPL projection.
# Usage: python inflow_post.py <inflowfit log files...>
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import minimize
import camb
from camb.dark_energy import DarkEnergyPPF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inflow_bg import de_history, BAO, BAO_DV, ZS_SN, S_SN, MNU, NS_MODEL, C_KMS, OM_LOCK, SIG_TH, SIG_OMM

def last_result(path):
    res = [l for l in open(path, encoding='utf-8') if l.startswith('RESULT ')]
    return json.loads(res[-1][7:]) if res else None

def growth(H0, ombh2, omch2, lnAs, tau, hist):
    p = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, mnu=MNU, As=np.exp(lnAs)*1e-10, ns=NS_MODEL, lmax=2500)
    if hist is not None:
        de = DarkEnergyPPF(); de.set_w_a_table(hist['a'], hist['w']); p.DarkEnergy = de
    p.set_matter_power(redshifts=[0.0, 0.4, 0.8], kmax=2.0)
    r = camb.get_results(p)
    s8 = r.get_sigma8_0(); fs8 = r.get_fsigma8()   # ordered by decreasing z? camb returns for redshifts sorted descending
    zs_sorted = sorted([0.0, 0.4, 0.8], reverse=True)
    return s8, dict(zip(zs_sorted, fs8)), r

def cpl_projection(r_model, h_m, om_m, kind_label):
    zs = np.array([BAO_DV[0]] + [b[0] for b in BAO])
    d = r_model.get_derived_params(); rd_m = d['rdrag']; th_m = d['thetastar']
    DM_m = r_model.comoving_radial_distance(zs); DH_m = C_KMS/np.array([r_model.hubble_parameter(z) for z in zs])
    MU_m = 5*np.log10((1+ZS_SN)*r_model.comoving_radial_distance(ZS_SN)); omm_m = om_m*h_m**2
    def chi2(x):
        om, h, w0, wa = x
        if not (0.15 < om < 0.5 and 0.5 < h < 0.9 and -2.5 < w0 < 0 and -5 < wa < 3 and w0 + wa < 0.5): return 1e9
        p = camb.set_params(H0=100*h, ombh2=0.02237, omch2=om*h*h - 0.02237 - MNU/93.14, mnu=MNU, tau=0.054, As=2.1e-9, ns=0.9656)
        de = DarkEnergyPPF(); de.set_params(w=w0, wa=wa); p.DarkEnergy = de
        try:
            r = camb.get_background(p); dd = r.get_derived_params()
        except Exception: return 1e9
        rd = dd['rdrag']; DM = r.comoving_radial_distance(zs); DH = C_KMS/np.array([r.hubble_parameter(z) for z in zs])
        x = (((zs[0]*DM[0]**2*DH[0])**(1/3)/rd - (zs[0]*DM_m[0]**2*DH_m[0])**(1/3)/rd_m)/BAO_DV[2])**2
        for i, (z, dmv, sm, dhv, sh, rho) in enumerate(BAO):
            dv = np.array([DM[i+1]/rd - DM_m[i+1]/rd_m, DH[i+1]/rd - DH_m[i+1]/rd_m]); cov = np.array([[sm*sm, rho*sm*sh], [rho*sm*sh, sh*sh]])
            x += dv @ np.linalg.inv(cov) @ dv
        mu = 5*np.log10((1+ZS_SN)*r.comoving_radial_distance(ZS_SN)); e = MU_m - mu; e -= e.mean(); x += float(np.sum((e/S_SN)**2))
        x += ((dd['thetastar'] - th_m)/SIG_TH)**2 + ((om*h*h - omm_m)/SIG_OMM)**2
        return x
    best = None
    for x0 in [(0.31, 0.68, -0.8, -0.8), (0.30, 0.69, -0.7, -1.2), (0.32, 0.67, -0.9, -0.4)]:
        res = minimize(chi2, np.array(x0), method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-3, maxfev=1200, adaptive=True))
        if best is None or res.fun < best.fun: best = res
    om, h, w0, wa = best.x; u = -(1+w0)/wa if wa != 0 else -1; zc = u/(1-u) if 0 < u < 1 else None
    return dict(Om=om, h=h, w0=w0, wa=wa, resid=best.fun, z_cross=zc)

for path in sys.argv[1:]:
    R = last_result(path)
    if R is None: print(path, 'no RESULT yet'); continue
    H0, ombh2, omch2, lnAs, tau = R['H0'], R['ombh2'], R['omch2'], R['lnAs'], R['tau']
    h = H0/100; om = (ombh2 + omch2 + MNU/93.14)/h**2
    hist = de_history(h, om, R['kind'], tuple(R['prm']))
    s8, fs8, r = growth(H0, ombh2, omch2, lnAs, tau, hist)
    print(f"\n=== {R['tag']}: total {R['total']:.2f} [CMB {R['CMB']:.2f} BAO {R['BAO']:.2f} SN {R['SN']:.2f}]  H0={H0:.2f} ombh2={ombh2:.5f} Om={om:.4f} tau={tau:.4f} nfev={R['nfev']} ({R['minutes']:.1f} min)")
    print(f"   inflow prm={[round(v,3) for v in R['prm']]}  F={hist['F']:.3f}  fM={hist['fM']:.3f}  birth reservoir/today = {hist['x_birth_ratio']:.3f}  z_cross={[round(v,2) for v in hist['z_cross']]}  w_min={hist['w_min']:.3f}  w0={hist['w0']:.3f}  t0={hist['t0']:.2f} Gyr")
    print(f"   sigma8={s8:.4f}  S8={s8*np.sqrt(om/0.3):.4f}  fsigma8: " + ', '.join(f'z={z}: {v:.4f}' for z, v in fs8.items()))
    print('   w_eff(z): ' + '  '.join(f"z={zz}:{hist['w'][np.argmin(abs(hist['z']-zz))]:+.3f}" for zz in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]))
    print('   rho_DE(z)/rho_DE(0): ' + '  '.join(f"z={zz}:{hist['x'][np.argmin(abs(hist['z']-zz))]/hist['x'][-1]:.3f}" for zz in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]))
    # time of switch-off / e-fold in Gyr and recursion-level units
    lev = hist['t0']/(2*1.6180339887**2)
    if R['kind'] == 'tophat':
        i = np.argmin(abs(hist['z'] - R['prm'][1])); print(f"   switch-off at z={R['prm'][1]:.3f}: t={hist['t'][i]:.2f} Gyr = {hist['t'][i]/lev:.2f} recursion levels (level length {lev:.2f} Gyr)")
    if R['kind'] == 'expdec':
        print(f"   accretion e-fold time {R['prm'][1]:.2f} Gyr = {R['prm'][1]/lev:.2f} recursion levels")
    cp = cpl_projection(r, h, om, R['tag'])
    print(f"   CPL projection (what a DESI-style analysis would infer): w0={cp['w0']:+.3f} wa={cp['wa']:+.3f} Om={cp['Om']:.4f} h={cp['h']:.4f} crossing z={cp['z_cross']}  (resid {cp['resid']:.2f})")
