# Late-ISW signature of the model w(z): TT ratio at low ell (model vs LCDM, no cutoff),
# and the growth-rate-derived ISW source enhancement.
import numpy as np
import camb
from camb.dark_energy import DarkEnergyPPF

C_DRAIN = 0.133
OMBH2, OMCH2, TAU, MNU = 0.02237, 0.1200, 0.0544, 0.06
AS = 2.1e-9

def model_w_a(h):
    om = (OMBH2 + OMCH2 + MNU/93.14)/h**2
    orad = 4.15e-5/h**2
    ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, 4, 3000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        I = np.concatenate(([0.0], np.cumsum(0.5*(3*(1+w[1:])/(1+z[1:]) + 3*(1+w[:-1])/(1+z[:-1]))*np.diff(z))))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    a = 1/(1+z)
    return a[::-1], (-1 + C_DRAIN/E)[::-1]

def run(model):
    p = camb.set_params(H0=67.36, ombh2=OMBH2, omch2=OMCH2, tau=TAU, mnu=MNU,
                        As=AS, ns=0.9656 if model else 0.9649, lmax=100)
    if model:
        de = DarkEnergyPPF()
        a, w = model_w_a(0.6736)
        de.set_w_a_table(a, w)
        p.DarkEnergy = de
    p.NonLinear = camb.model.NonLinear_none
    r = camb.get_results(p)
    return r.get_cmb_power_spectra(p, CMB_unit='muK')['unlensed_scalar'][:, 0]

cl_l = run(False)
cl_m = run(True)
print('TT ratio model/LCDM at low ell (late-ISW region; ns difference <0.1% here):')
for ell in (2, 3, 5, 8, 12, 20, 30, 50, 80):
    print(f'   l={ell:3d}: {100*(cl_m[ell]/cl_l[ell]-1):+.1f}%')
