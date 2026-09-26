# Joint fit: Planck 2018 compressed distance priors (R, l_A, omega_b; Chen-Huang-Wang 2018)
# + DESI DR2 BAO (13 points, per-tracer DM-DH correlations).
# Cases: (1) LCDM free; (2) model locked (Omega_m=0.312, w(z), ns fixed);
#        (3) model + free N_eff; (4) model + free Y_He.
# Background-only CAMB per point (fast). Note: DESI DR2 central values from the DR2 paper
# as recorded here; small transcription deviations do not change conclusions.
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
import camb
from camb.dark_energy import DarkEnergyPPF

C_KMS = 299792.458
C_DRAIN = 0.133
MNU = 0.06

# ---- Planck 2018 distance priors (TT,TE,EE+lowE), Chen-Huang-Wang 1808.05724 ----
PRI_MEAN = np.array([1.7502, 301.471, 0.02236])
PRI_SIG  = np.array([0.0046, 0.090, 0.00015])
PRI_CORR = np.array([[1.0, 0.46, -0.66],
                     [0.46, 1.0, -0.33],
                     [-0.66, -0.33, 1.0]])
PRI_COV = PRI_CORR * np.outer(PRI_SIG, PRI_SIG)
PRI_ICOV = np.linalg.inv(PRI_COV)

# ---- DESI DR2 BAO ----
# (z, DM/rd, sig, DH/rd, sig, corr) ; BGS is DV-only
BAO_DV = (0.295, 7.944, 0.075)
BAO = [
    (0.510, 13.588, 0.167, 21.863, 0.427, -0.459),
    (0.706, 17.351, 0.177, 19.455, 0.330, -0.404),
    (0.934, 21.576, 0.152, 17.641, 0.193, -0.416),
    (1.321, 27.601, 0.318, 14.176, 0.221, -0.434),
    (1.484, 30.512, 0.760, 12.817, 0.516, -0.500),
    (2.330, 38.988, 0.531,  8.632, 0.101, -0.431),
]

def model_w_a(h, om, orad):
    ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, 4, 2000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        I = np.concatenate(([0.0], cumulative_trapezoid(3*(1+w)/(1+z), z)))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    a = 1/(1+z)
    return a[::-1], (-1 + C_DRAIN/E)[::-1]

def background(ombh2, omch2, h, nnu=3.044, yhe=None, model=False):
    kw = dict(H0=100*h, ombh2=ombh2, omch2=omch2, mnu=MNU, nnu=nnu,
              As=2.1e-9, ns=0.9656, tau=0.054)
    p = camb.set_params(**kw)
    if yhe is not None:
        p.YHe = yhe
    if model:
        om = (ombh2 + omch2 + MNU/93.14)/h**2
        orad = 2.4728e-5*(1 + 0.2271*nnu)/h**2
        a, w = model_w_a(h, om, orad)
        de = DarkEnergyPPF(); de.set_w_a_table(a, w)
        p.DarkEnergy = de
    r = camb.get_background(p)
    d = r.get_derived_params()
    zstar, rstar, rdrag = d['zstar'], d['rstar'], d['rdrag']
    DMstar = r.comoving_radial_distance(zstar)
    om_tot = (ombh2 + omch2 + MNU/93.14)/h**2
    R = np.sqrt(om_tot)*(100*h/C_KMS)*DMstar
    lA = np.pi*DMstar/rstar
    # BAO predictions
    zs = np.array([BAO_DV[0]] + [b[0] for b in BAO])
    DM = r.comoving_radial_distance(zs)
    Hz = np.array([r.hubble_parameter(z) for z in zs])
    DH = C_KMS/Hz
    return R, lA, rdrag, DM, DH, d

def chi2_total(R, lA, ombh2, rdrag, DM, DH):
    v = np.array([R, lA, ombh2]) - PRI_MEAN
    x2 = v @ PRI_ICOV @ v
    # BGS DV
    z0 = BAO_DV[0]
    dv = (z0*DM[0]**2*DH[0])**(1/3)/rdrag
    x2 += ((dv - BAO_DV[1])/BAO_DV[2])**2
    for i, (z, dmv, sm, dhv, sh, rho) in enumerate(BAO):
        dm_p = DM[i+1]/rdrag; dh_p = DH[i+1]/rdrag
        dvec = np.array([dm_p - dmv, dh_p - dhv])
        cov = np.array([[sm*sm, rho*sm*sh], [rho*sm*sh, sh*sh]])
        x2 += dvec @ np.linalg.inv(cov) @ dvec
    return x2

def run_case(name, x0, unpack, model, extra=lambda p: {}):
    def f(p):
        try:
            ombh2, omch2, h, kw = unpack(p)
            if not (0.019 < ombh2 < 0.026 and 0.05 < omch2 < 0.25 and 0.55 < h < 0.80):
                return 1e10
            R, lA, rdrag, DM, DH, d = background(ombh2, omch2, h, model=model, **kw)
            return chi2_total(R, lA, ombh2, rdrag, DM, DH)
        except Exception:
            return 1e10
    r = minimize(f, x0, method='Nelder-Mead',
                 options=dict(xatol=1e-5, fatol=1e-4, maxiter=2500, maxfev=2500))
    ombh2, omch2, h, kw = unpack(r.x)
    R, lA, rdrag, DM, DH, d = background(ombh2, omch2, h, model=model, **kw)
    om = (ombh2 + omch2 + MNU/93.14)/h**2
    print(f'{name}')
    print(f'   chi2 = {r.fun:7.2f}   ombh2 = {ombh2:.5f}  omch2 = {omch2:.5f}  H0 = {100*h:.2f}  Om = {om:.4f}'
          + ''.join(f'  {k} = {v:.4f}' for k, v in kw.items()))
    print(f'   R = {R:.4f}  l_A = {lA:.3f}  r_drag = {rdrag:.2f} Mpc  100theta* = {d["thetastar"]:.5f}')
    return r.fun

print('data: Planck-2018 distance priors (R, l_A, omega_b) + DESI DR2 BAO (13 pts)')
print()
c1 = run_case('case 1: LCDM (ombh2, omch2, h free)',
              [0.02237, 0.1200, 0.6736],
              lambda p: (p[0], p[1], p[2], {}), model=False)
print()
def unpack_locked(p, nnu=3.044, yhe=None):
    ombh2, h = p[0], p[1]
    if not (3.044 <= nnu <= 4.5): raise ValueError('nnu out of physical relic range')
    if yhe is not None and not (0.18 <= yhe <= 0.32): raise ValueError('yhe out of range')
    omch2 = 0.312*h*h - ombh2 - MNU/93.14
    kw = {}
    if nnu != 3.044: kw['nnu'] = nnu
    if yhe is not None: kw['yhe'] = yhe
    return ombh2, omch2, h, kw
c2 = run_case('case 2: model locked (Om=0.312, w(z); ombh2, h free)',
              [0.02237, 0.677],
              lambda p: unpack_locked(p), model=True)
print()
c3 = None
for start in ([0.02237, 0.690, 3.60], [0.02237, 0.683, 3.30], [0.02250, 0.695, 4.00]):
    v = run_case(f'case 3: model + free N_eff >= 3.044 (start nnu={start[2]})',
                 start, lambda p: unpack_locked(p[:2], nnu=p[2]), model=True)
    c3 = v if c3 is None else min(c3, v)
    print()
print()
c4 = run_case('case 4: model + free Y_He in [0.18, 0.32]',
              [0.02237, 0.675, 0.28],
              lambda p: unpack_locked(p[:2], yhe=p[2]), model=True)
print()
print(f'delta-chi2 vs LCDM:  model locked {c2-c1:+.1f}   model+Neff {c3-c1:+.1f}   model+YHe {c4-c1:+.1f}')
