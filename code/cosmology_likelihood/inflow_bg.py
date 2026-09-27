import os, sys, json, time
# Inflow door: drain + parent-inflow source in the DE continuity equation.
#   d rho_DE/dt = -Gamma rho_DE + S(t),  Gamma = 3 C H0 (committed drain, C = 0.133)
# Backward integration from today (rho_DE,0 fixed by flatness with locked Omega_m = 0.312).
# Families for S (dimensionless s = S/(rho_crit0 H0)):
#   'tophat': s = A for z > z_end (constant accretion in cosmic time, switched off at z_end)
#   'expdec': s = A exp(-t/t_acc) (front-loaded accretion fading with e-fold time t_acc, Gyr)
#   'gauss' : s = A exp(-(N-N_c)^2/(2 sig^2)), N = ln a (an accretion episode)
# Background proxy for the CMB: Gaussian priors on omega_m (0.1430+-0.0011) and 100 theta* (1.04110+-0.00031),
# omega_b fixed at 0.02237, h profiled (Omega_m locked at 0.312 => omega_m = 0.312 h^2).
# Usage: python inflow_bg.py ref | scan <family> | point <family> <params...>
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid
import camb
from camb.dark_energy import DarkEnergyPPF

C_DRAIN, MNU, NS_MODEL, C_KMS = 0.133, 0.06, 0.9656, 299792.458
GAMMA = 3*C_DRAIN
OM_LOCK = 0.312
OMBH2 = 0.02237
TH_PL, SIG_TH = 1.04110, 0.00031          # 100 theta*
OMM_PL, SIG_OMM = 0.1430, 0.0011
NGRID = 2000
N_GRID = np.linspace(0.0, np.log(1e-4), NGRID)
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

def de_history(h, om, kind, prm):
    orad = 4.15e-5/h**2
    x0 = 1 - om - orad
    tH = 977.8/(100*h)                      # 1/H0 in Gyr
    a_grid = np.exp(N_GRID); z_grid = 1/a_grid - 1
    def hub(N, x):
        a = np.exp(N)
        return np.sqrt(om*a**-3 + orad*a**-4 + max(x, 1e-12))
    def source(N, t):
        if kind in ('none', 'lcdm'): return 0.0
        if kind == 'tophat':
            A, z_end = prm
            return A if (np.exp(-N) - 1) > z_end else 0.0
        if kind == 'expdec':
            A, t_acc = prm
            return A*np.exp(-t/t_acc)
        if kind == 'gauss':
            A, zc, sig = prm
            return A*np.exp(-(N + np.log(1+zc))**2/(2*sig**2))
        raise ValueError(kind)
    dN = N_GRID[1] - N_GRID[0]
    def time_table(hh):
        lb = np.concatenate(([0.0], np.cumsum(0.5*(1/hh[1:] + 1/hh[:-1])*(-dN))))
        t0 = lb[-1] + (np.exp(N_GRID[-1])**2)/(2*np.sqrt(orad))
        return (t0 - lb)*tH, t0*tH
    def integrate(tN):
        x = np.empty(NGRID); s_arr = np.empty(NGRID); hh = np.empty(NGRID)
        x[0] = x0
        for i in range(NGRID-1):
            N = N_GRID[i]; xi = x[i]
            t_i = tN[i]; t_h = 0.5*(tN[i]+tN[i+1]); t_n = tN[i+1]
            s_i = source(N, t_i); s_h = source(N+0.5*dN, t_h); s_n = source(N+dN, t_n)
            k1 = (-GAMMA*xi + s_i)/hub(N, xi)
            y = xi + 0.5*dN*k1; k2 = (-GAMMA*y + s_h)/hub(N+0.5*dN, y)
            y = xi + 0.5*dN*k2; k3 = (-GAMMA*y + s_h)/hub(N+0.5*dN, y)
            y = xi + dN*k3;     k4 = (-GAMMA*y + s_n)/hub(N+dN, y)
            x[i+1] = xi + dN*(k1+2*k2+2*k3+k4)/6
            s_arr[i] = s_i; hh[i] = hub(N, xi)
        s_arr[-1] = source(N_GRID[-1], tN[-1]); hh[-1] = hub(N_GRID[-1], x[-1])
        return x, s_arr, hh
    hh0 = np.sqrt(om*a_grid**-3 + orad*a_grid**-4 + x0)
    tN, t0 = time_table(hh0)
    passes = 3 if kind == 'expdec' else 1
    for _ in range(passes):
        x, s_arr, hh = integrate(tN)
        tN, t0 = time_table(hh)
    if np.min(x) <= 1e-6:
        return None
    w = -1 + GAMMA/(3*hh) - s_arr/(3*hh*x)
    if np.min(w) < -6 or np.max(w) > 0.5:
        return None
    t_birth = tN[-1]
    F = 1 - x[-1]*np.exp(-GAMMA*(t0 - t_birth)/tH)/x0
    integrand = s_arr*a_grid**3/hh
    fM = float(np.sum(0.5*(integrand[1:] + integrand[:-1])*(-dN)))
    cross = np.where(np.diff(np.sign(w + 1)) != 0)[0]
    z_cross = [float(z_grid[i]) for i in cross]
    return dict(a=a_grid[::-1], w=w[::-1], x=x[::-1], z=z_grid[::-1], t=tN[::-1], t0=t0, F=float(F), fM=fM, z_cross=z_cross,
                w_min=float(np.min(w)), w0=float(w[0]), x_birth_ratio=float(x[-1]/x0))

def background(h, kind, prm, ombh2=OMBH2, om=OM_LOCK):
    omch2 = om*h*h - ombh2 - MNU/93.14
    hist = de_history(h, om, kind, prm)
    if hist is None: return None, None
    p = camb.set_params(H0=100*h, ombh2=ombh2, omch2=omch2, mnu=MNU, tau=0.054, As=2.1e-9, ns=NS_MODEL)
    if kind != 'lcdm':
        de = DarkEnergyPPF(); de.set_w_a_table(hist['a'], hist['w']); p.DarkEnergy = de
    r = camb.get_background(p)
    d = r.get_derived_params()
    rd = d['rdrag']
    zs = np.array([BAO_DV[0]] + [b[0] for b in BAO])
    DM = r.comoving_radial_distance(zs); DH = C_KMS/np.array([r.hubble_parameter(zz) for zz in zs])
    x_bao = (((zs[0]*DM[0]**2*DH[0])**(1/3)/rd - BAO_DV[1])/BAO_DV[2])**2
    for i, (zz, dmv, sm, dhv, sh, rho) in enumerate(BAO):
        dvec = np.array([DM[i+1]/rd - dmv, DH[i+1]/rd - dhv]); cov = np.array([[sm*sm, rho*sm*sh], [rho*sm*sh, sh*sh]])
        x_bao += dvec @ np.linalg.inv(cov) @ dvec
    mu = 5*np.log10((1+ZS_SN)*r.comoving_radial_distance(ZS_SN)); dd = MU_SN - mu; dd -= dd.mean()
    x_sn = float(np.sum((dd/S_SN)**2))
    x_cmb = ((d['thetastar'] - TH_PL)/SIG_TH)**2 + ((om*h*h - OMM_PL)/SIG_OMM)**2
    out = dict(h=h, thetastar=d['thetastar'], dtheta_sig=(d['thetastar'] - TH_PL)/SIG_TH, rdrag=rd, x_bao=float(x_bao), x_sn=x_sn,
               x_cmb=float(x_cmb), F=hist['F'], fM=hist['fM'], z_cross=hist['z_cross'], w_min=hist['w_min'], w0=hist['w0'],
               t0=hist['t0'], x_birth_ratio=hist['x_birth_ratio'], DM=DM.tolist(), DH=DH.tolist())
    out['tot'] = out['x_cmb'] + out['x_bao'] + out['x_sn']
    return out, hist

def profile_h(kind, prm, what='tot'):
    cache = {}
    def f(h):
        o, _ = background(h, kind, prm)
        if o is None: return 1e6
        cache[h] = o
        return o[what]
    res = minimize_scalar(f, bounds=(0.60, 0.74), method='bounded', options=dict(xatol=2e-4))
    if res.fun >= 1e6: return None
    return cache[res.x]

def fmt(r):
    return (f"h={r['h']:.4f} dth={r['dtheta_sig']:+6.2f}s CMBp={r['x_cmb']:6.2f} BAO={r['x_bao']:6.2f} SN={r['x_sn']:5.2f} tot={r['tot']:6.2f} "
            f"F={r['F']:.3f} fM={r['fM']:.3f} zx={[round(v,2) for v in r['z_cross']]} wmin={r['w_min']:.2f} w0={r['w0']:.3f}")

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'ref':
        for kind, prm in [('lcdm', ()), ('none', ())]:
            r = profile_h(kind, prm); print(kind, '(profiled h, tot):', fmt(r))
            r = profile_h(kind, prm, 'x_cmb'); print(kind, '(profiled h, CMB proxy only):', fmt(r))
    elif mode == 'scan':
        kind = sys.argv[2]
        t0 = time.time(); rows = []
        if kind == 'tophat':
            grid = [(A, ze) for A in [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0] for ze in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0]]
        elif kind == 'expdec':
            grid = [(A, ta) for A in [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0] for ta in [1.0, 1.5, 2.0, 2.64, 3.5, 5.0, 7.0, 10.0, 14.0]]
        elif kind == 'gauss':
            grid = [(A, zc, sg) for A in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0] for zc in [0.5, 0.7, 1.0, 1.4, 2.0, 3.0] for sg in [0.25, 0.5, 0.8]]
        for prm in grid:
            r = profile_h(kind, prm)
            if r is None:
                rows.append(dict(prm=prm, ok=False)); print('  unphysical', prm); continue
            r['prm'] = prm; r['ok'] = True; rows.append(r)
            print(f"  prm={tuple(round(v,3) for v in prm)}  {fmt(r)}", flush=True)
        rows_ok = sorted([r for r in rows if r['ok']], key=lambda r: r['tot'])
        print(f'\n{kind}: {len(rows_ok)}/{len(rows)} physical; {time.time()-t0:.0f}s')
        print('best 12 by total (CMB proxy + BAO + SN):')
        for r in rows_ok[:12]: print(f"  prm={tuple(round(v,3) for v in r['prm'])}  {fmt(r)}")
        json.dump(rows, open(f'inflow_scan_{kind}.json', 'w'), default=float)
    elif mode == 'point':
        kind = sys.argv[2]; prm = tuple(float(v) for v in sys.argv[3:])
        r = profile_h(kind, prm); print(fmt(r))
        o, hist = background(r['h'], kind, prm)
        for zz in [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
            i = np.argmin(abs(hist['z'] - zz))
            print(f"  z={zz:5.1f}  w_eff={hist['w'][i]:+7.3f}  rho_DE/rho_DE0={hist['x'][i]/hist['x'][0]:6.3f}  t={hist['t'][i]:6.2f} Gyr")
