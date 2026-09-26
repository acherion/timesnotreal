# Verify the w(a) table orientation and CAMB's background under the model DE.
import numpy as np
import camb
from camb.dark_energy import DarkEnergyPPF

C_DRAIN = 0.133
OMBH2, OMCH2, TAU, MNU = 0.02237, 0.1200, 0.0544, 0.06
AS, NS = 2.1e-9, 0.9656

def model_w_a(h):
    om = (OMBH2 + OMCH2 + MNU/93.14) / h**2
    orad = 4.15e-5 / h**2
    ode0 = 1 - om - orad
    z = np.logspace(-4, 4, 3000)
    zg = np.concatenate(([0.0], z))
    E = np.sqrt(om*(1+zg)**3 + orad*(1+zg)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        integ = 3*(1+w)/(1+zg)
        I = np.concatenate(([0.0], np.cumsum(0.5*(integ[1:]+integ[:-1])*np.diff(zg))))
        E = np.sqrt(om*(1+zg)**3 + orad*(1+zg)**4 + ode0*np.exp(I))
    a = 1/(1+zg)
    return a[::-1], (-1 + C_DRAIN/E)[::-1], zg, E

h = 0.6736
a, w, zg, E_py = model_w_a(h)
print(f'w(a=1)   = {w[-1]:.4f}   (must be ~ -0.867)')
print(f'w(a=0.5) = {np.interp(0.5, a, w):.4f}   (z=1, expect ~ -0.93)')
print(f'w(a=0.1) = {np.interp(0.1, a, w):.4f}   (z=9, expect ~ -0.99)')

p = camb.set_params(H0=100*h, ombh2=OMBH2, omch2=OMCH2, tau=TAU, mnu=MNU, As=AS, ns=NS)
de = DarkEnergyPPF(); de.set_w_a_table(a, w); p.DarkEnergy = de
r = camb.get_results(p)

# compare camb H(z) against the python integration
print('\nH(z) cross-check (camb vs standalone integration):')
H0 = r.hubble_parameter(0.0)
for z in (0.0, 0.5, 1.0, 2.0, 5.0):
    Hc = r.hubble_parameter(z)/H0
    Hp = np.interp(z, zg, E_py)
    print(f'  z={z:4.1f}:  E_camb = {Hc:.4f}   E_python = {Hp:.4f}   ratio = {Hc/Hp:.5f}')

# sanity: constant w=-0.9 shift of theta* (known-direction control)
d_model = r.get_derived_params()
p2 = camb.set_params(H0=100*h, ombh2=OMBH2, omch2=OMCH2, tau=TAU, mnu=MNU, As=AS, ns=NS)
p2.set_dark_energy(w=-0.9, wa=0, dark_energy_model='ppf')
d_w9 = camb.get_results(p2).get_derived_params()
p3 = camb.set_params(H0=100*h, ombh2=OMBH2, omch2=OMCH2, tau=TAU, mnu=MNU, As=AS, ns=NS)
d_lcdm = camb.get_results(p3).get_derived_params()
print(f"\n100theta*: LCDM = {d_lcdm['thetastar']:.5f}, const w=-0.9 = {d_w9['thetastar']:.5f} "
      f"({100*(d_w9['thetastar']/d_lcdm['thetastar']-1):+.2f}%), model w(z) = {d_model['thetastar']:.5f} "
      f"({100*(d_model['thetastar']/d_lcdm['thetastar']-1):+.2f}%)")
