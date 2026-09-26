# Growth through the real pipeline: fsigma8(z) for LCDM vs the model (CAMB), against a
# 13-point RSD compilation; S8 at the model's corners; physical lensing amplitude A_L.
import numpy as np
import camb
from camb.dark_energy import DarkEnergyPPF

C_DRAIN = 0.133
OMBH2, TAU, MNU, AS = 0.02237, 0.0544, 0.06, 2.1e-9

RSD = [  # (z, fsigma8, sigma, survey)
    (0.067, 0.423, 0.055, '6dFGS'), (0.15, 0.53, 0.16, 'SDSS MGS'),
    (0.38, 0.497, 0.045, 'BOSS DR12'), (0.51, 0.458, 0.038, 'BOSS DR12'), (0.61, 0.436, 0.034, 'BOSS DR12'),
    (0.44, 0.413, 0.080, 'WiggleZ'), (0.60, 0.390, 0.063, 'WiggleZ'), (0.73, 0.437, 0.072, 'WiggleZ'),
    (0.70, 0.473, 0.041, 'eBOSS LRG'), (0.85, 0.315, 0.095, 'eBOSS ELG'), (1.48, 0.462, 0.045, 'eBOSS QSO'),
    (0.86, 0.40, 0.11, 'VIPERS'), (1.40, 0.482, 0.116, 'FastSound'),
]
zs = np.array([r[0] for r in RSD]); fs8 = np.array([r[1] for r in RSD]); sig = np.array([r[2] for r in RSD])

def model_w_a(h, om, orad):
    ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, 4, 3000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        I = np.concatenate(([0.0], np.cumsum(0.5*(3*(1+w[1:])/(1+z[1:]) + 3*(1+w[:-1])/(1+z[:-1]))*np.diff(z))))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    return (1/(1+z))[::-1], (-1 + C_DRAIN/E)[::-1]

def run(h0, omch2, model, ns, zlist):
    p = camb.set_params(H0=h0, ombh2=OMBH2, omch2=omch2, tau=TAU, mnu=MNU, As=AS, ns=ns, lmax=2500)
    h = h0/100
    if model:
        om = (OMBH2 + omch2 + MNU/93.14)/h**2
        a, w = model_w_a(h, om, 4.15e-5/h**2)
        de = DarkEnergyPPF(); de.set_w_a_table(a, w); p.DarkEnergy = de
    p.set_matter_power(redshifts=sorted(set(list(zlist) + [0.0])), kmax=2.0)
    p.NonLinear = camb.model.NonLinear_none
    r = camb.get_results(p)
    zr = np.array(r.transfer_redshifts)     # descending
    f8 = np.array(r.get_fsigma8())          # same order as transfer redshifts
    s8 = np.array(r.get_sigma8())
    order = np.argsort(zr)
    zr, f8, s8 = zr[order], f8[order], s8[order]
    om = (OMBH2 + omch2 + MNU/93.14)/h**2
    return dict(z=zr, fs8=f8, s8=s8, sigma8_0=r.get_sigma8_0(), S8=r.get_sigma8_0()*np.sqrt(om/0.3), res=r, pars=p, om=om)

def chi2(run_out):
    pred = np.interp(zs, run_out['z'], run_out['fs8'])
    return np.sum(((pred - fs8)/sig)**2), pred

zgrid = sorted(set(list(zs) + [0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.2, 2.0]))

print('=== fsigma8 vs 13-point RSD compilation (diagonal errors) ===')
L = run(67.36, 0.1200, False, 0.9649, zgrid)
M1 = run(67.36, 0.1200, True, 0.9656, zgrid)                      # model, Planck params (isolates w(z) effect)
h2 = 0.681; M2 = run(68.1, 0.312*h2*h2 - OMBH2 - MNU/93.14, True, 0.9656, zgrid)   # model, own corner
h3 = 0.6664; M3 = run(66.64, 0.312*h3*h3 - OMBH2 - MNU/93.14, True, 0.9656, zgrid)  # model, CMB-distance corner
for name, R in (('LCDM (Planck)', L), ('model @ Planck params', M1), ('model @ own (68.1, Om=0.312)', M2), ('model @ CMB corner (66.6, Om=0.312)', M3)):
    x2, pred = chi2(R)
    print(f'{name:36s}: chi2 = {x2:5.2f} / 13 pts   sigma8 = {R["sigma8_0"]:.4f}   S8 = {R["S8"]:.4f}')
print()
print('fsigma8 suppression, model@Planck vs LCDM:')
for z in (0.1, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5):
    fl = np.interp(z, L['z'], L['fs8']); fm = np.interp(z, M1['z'], M1['fs8'])
    print(f'   z={z:3.1f}: {100*(fm/fl-1):+.1f}%')
print()
# physical lensing amplitude: ratio of lensing potential power spectra C_l^{phiphi}
cl_L = L['res'].get_lens_potential_cls(lmax=2000)[:, 0]
cl_M = M1['res'].get_lens_potential_cls(lmax=2000)[:, 0]
print('physical A_L proxy (C_l^phiphi model/LCDM at Planck params):')
for ell in (40, 100, 200, 400, 800):
    print(f'   l={ell:4d}: {cl_M[ell]/cl_L[ell]:.4f}')
lo, hi = 40, 400
print(f'   band-average l={lo}-{hi}: {np.mean(cl_M[lo:hi]/cl_L[lo:hi]):.4f}   (site claims ~0.95)')
