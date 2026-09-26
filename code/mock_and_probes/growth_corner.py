import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
src = open(r'growth.py', encoding='utf-8').read()
exec(src.split("print('=== fsigma8")[0])
import numpy as np, camb
from camb.dark_energy import DarkEnergyPPF
def run2(h0, ombh2, omch2, lnAs, zlist):
    p = camb.set_params(H0=h0, ombh2=ombh2, omch2=omch2, tau=0.0544, mnu=MNU, As=np.exp(lnAs)*1e-10, ns=0.9656, lmax=2500)
    hh = h0/100; om = (ombh2 + omch2 + MNU/93.14)/hh**2
    a, w = model_w_a(hh, om, 4.15e-5/hh**2); de = DarkEnergyPPF(); de.set_w_a_table(a, w); p.DarkEnergy = de
    p.set_matter_power(redshifts=sorted(set(list(zlist) + [0.0])), kmax=2.0); p.NonLinear = camb.model.NonLinear_none
    r = camb.get_results(p); zr = np.array(r.transfer_redshifts); f8 = np.array(r.get_fsigma8()); o = np.argsort(zr)
    return dict(z=zr[o], fs8=f8[o], sigma8_0=r.get_sigma8_0(), S8=r.get_sigma8_0()*np.sqrt(om/0.3))
h = 0.6652; omc = 0.312*h*h - 0.02278 - MNU/93.14
R = run2(66.52, 0.02278, omc, 3.0351, zgrid)
x2, pred = chi2(R)
print(f'model @ CMB best fit (66.5, Om=0.312, lnAs=3.035): RSD chi2 = {x2:.2f} / 13   sigma8 = {R["sigma8_0"]:.4f}   S8 = {R["S8"]:.4f}')
