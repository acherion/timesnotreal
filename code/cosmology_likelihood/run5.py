# Run 5: the model at its OWN parameters (H0=68.1, Omega_m=0.312) vs measured theta*
import numpy as np
import camb
from camb.dark_energy import DarkEnergyPPF

C_DRAIN = 0.133
NS_MODEL = 0.9656
AS = 2.1e-9
OMBH2, TAU, MNU = 0.02237, 0.0544, 0.06

def model_w_a(h, om, orad, ode0):
    z = np.logspace(-4, 4, 3000)
    zg = np.concatenate(([0.0], z))
    E = np.sqrt(om*(1+zg)**3 + orad*(1+zg)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        integ = 3*(1+w)/(1+zg)
        I = np.concatenate(([0.0], np.cumsum(0.5*(integ[1:]+integ[:-1])*np.diff(zg))))
        E = np.sqrt(om*(1+zg)**3 + orad*(1+zg)**4 + ode0*np.exp(I))
    a = 1/(1+zg)
    return a[::-1], (-1 + C_DRAIN/E)[::-1]

h = 0.681
om = 0.312
orad = 4.15e-5/h**2
omch2_own = om*h*h - OMBH2 - MNU/93.14
p = camb.set_params(H0=100*h, ombh2=OMBH2, omch2=omch2_own, tau=TAU, mnu=MNU,
                    As=AS, ns=NS_MODEL, lmax=2600)
a, w = model_w_a(h, om, orad, 1-om-orad)
de = DarkEnergyPPF(); de.set_w_a_table(a, w)
p.DarkEnergy = de
p.set_matter_power(redshifts=[0.0], kmax=2.0)
r = camb.get_results(p)
d = r.get_derived_params()
chi = r.comoving_radial_distance(d['zstar'])
R = np.sqrt(om)*(100*h/299792.458)*chi
s8 = r.get_sigma8_0()
print(f"model self-consistent (H0=68.1, Om=0.312):")
print(f"  100theta* = {d['thetastar']:.5f}   (Planck measured 1.04109 +- 0.00030  ->  {(d['thetastar']-1.04109)/0.00030:+.0f} sigma)")
print(f"  R = {R:.4f}  ({(R-1.7502)/0.0046:+.1f} sigma vs 1.7502+-0.0046)")
print(f"  sigma8 = {s8:.4f}   S8 = {s8*np.sqrt(om/0.3):.4f}")
print(f"  age = {d['age']:.2f} Gyr   H0*t0 = {100*h*d['age']/978.0:.4f}")
