import os, sys, json
# Mock-DESI test: if the inflow universe were true, what (w0, wa) would a CPL analysis of DESI-DR2-like BAO
# + CMB distance priors + SN shape infer?  Compare with DESI DR2 real results.
# Usage: python inflow_mockcpl.py <family> <params...>   (h is profiled by the background proxy as in inflow_bg)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import minimize
import camb
from camb.dark_energy import DarkEnergyPPF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inflow_bg import background, profile_h, BAO, BAO_DV, ZS_SN, S_SN, MNU, C_KMS, OMBH2, SIG_TH, SIG_OMM

kind = sys.argv[1]; prm = tuple(float(v) for v in sys.argv[2:])
ref = profile_h(kind, prm)
o, hist = background(ref['h'], kind, prm)
h_m = ref['h']; om_m = 0.312
zs = np.array([BAO_DV[0]] + [b[0] for b in BAO])
DM_m = np.array(o['DM']); DH_m = np.array(o['DH']); rd_m = o['rdrag']; th_m = o['thetastar']; omm_m = om_m*h_m**2
# mock SN shape: model distance modulus (shape only)
p = camb.set_params(H0=100*h_m, ombh2=OMBH2, omch2=om_m*h_m**2 - OMBH2 - MNU/93.14, mnu=MNU, tau=0.054, As=2.1e-9, ns=0.9656)
de = DarkEnergyPPF(); de.set_w_a_table(hist['a'], hist['w']); p.DarkEnergy = de
r = camb.get_background(p); MU_m = 5*np.log10((1+ZS_SN)*r.comoving_radial_distance(ZS_SN))
print(f'inflow universe: {kind} {prm}  h={h_m:.4f}  F={o["F"]:.3f} fM={o["fM"]:.3f}  z_cross={o["z_cross"]}  w_min={o["w_min"]:.3f} w0={o["w0"]:.3f}')
for zz in [0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    i = np.argmin(abs(hist['z'] - zz)); print(f'   z={zz:4.1f}  w_eff={hist["w"][i]:+.3f}  rho/rho0={hist["x"][i]/hist["x"][0]:.3f}')

def cpl_chi2(x, return_parts=False):
    om, h, w0, wa = x
    if not (0.15 < om < 0.5 and 0.5 < h < 0.9 and -2.5 < w0 < 0 and -5 < wa < 3 and w0 + wa < 0.5): return 1e9
    p = camb.set_params(H0=100*h, ombh2=OMBH2, omch2=om*h*h - OMBH2 - MNU/93.14, mnu=MNU, tau=0.054, As=2.1e-9, ns=0.9656)
    de = DarkEnergyPPF(); de.set_params(w=w0, wa=wa); p.DarkEnergy = de
    try:
        r = camb.get_background(p); d = r.get_derived_params()
    except Exception:
        return 1e9
    rd = d['rdrag']
    DM = r.comoving_radial_distance(zs); DH = C_KMS/np.array([r.hubble_parameter(z) for z in zs])
    x_bao = (((zs[0]*DM[0]**2*DH[0])**(1/3)/rd - (zs[0]*DM_m[0]**2*DH_m[0])**(1/3)/rd_m)/BAO_DV[2])**2
    for i, (z, dmv, sm, dhv, sh, rho) in enumerate(BAO):
        dvec = np.array([DM[i+1]/rd - DM_m[i+1]/rd_m, DH[i+1]/rd - DH_m[i+1]/rd_m]); cov = np.array([[sm*sm, rho*sm*sh], [rho*sm*sh, sh*sh]])
        x_bao += dvec @ np.linalg.inv(cov) @ dvec
    mu = 5*np.log10((1+ZS_SN)*r.comoving_radial_distance(ZS_SN)); dd = MU_m - mu; dd -= dd.mean(); x_sn = float(np.sum((dd/S_SN)**2))
    x_cmb = ((d['thetastar'] - th_m)/SIG_TH)**2 + ((om*h*h - omm_m)/SIG_OMM)**2
    tot = x_bao + x_sn + x_cmb
    if return_parts: return tot, dict(bao=float(x_bao), sn=x_sn, cmb=float(x_cmb))
    return tot

best = None
for x0 in [(0.31, 0.68, -0.8, -0.8), (0.30, 0.69, -0.7, -1.2), (0.32, 0.67, -0.9, -0.4), (0.31, 0.68, -1.0, 0.0)]:
    res = minimize(cpl_chi2, np.array(x0), method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-3, maxfev=1500, adaptive=True))
    if best is None or res.fun < best.fun: best = res
om, h, w0, wa = best.x
tot, parts = cpl_chi2(best.x, True)
zc = None
if wa != 0:
    u = -(1 + w0)/wa
    if 0 < u < 1: zc = u/(1-u)
print(f'CPL an analyst would infer: Om={om:.4f} h={h:.4f} w0={w0:+.3f} wa={wa:+.3f}  (residual chi2 {tot:.2f}: {parts})  crossing z={zc}')
print('DESI DR2 real (BAO+CMB+SN): DESY5 (-0.752, -0.86) | Pantheon+ (-0.838, -0.62) | Union3 (-0.667, -1.09); crossings z~0.4-0.5')
# also: LCDM an analyst would infer (wa=w0+1=0) and its residual
def lcdm_chi2(x): return cpl_chi2((x[0], x[1], -1.0, 0.0))
res = minimize(lcdm_chi2, np.array([0.31, 0.68]), method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-3, adaptive=True))
tot_l, parts_l = cpl_chi2((res.x[0], res.x[1], -1.0, 0.0), True)
print(f'LCDM an analyst would infer: Om={res.x[0]:.4f} h={res.x[1]:.4f}  residual chi2 {tot_l:.2f}: {parts_l}  -> LCDM would be disfavoured by dchi2={tot_l-tot:.1f} (2 dof) in this mock')
json.dump(dict(kind=kind, prm=prm, h_model=h_m, cpl=dict(Om=om, h=h, w0=w0, wa=wa, chi2=tot, parts=parts, z_cross=zc), lcdm=dict(Om=res.x[0], h=res.x[1], chi2=tot_l)),
          open(f'inflow_mockcpl_{kind}.json', 'w'), default=float)
