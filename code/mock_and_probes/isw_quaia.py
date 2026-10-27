import os, sys
# ISW-galaxy cross-correlation amplitude relative to Planck LCDM at the Quaia tomographic bins
# (Ghodsi Yengejeh et al. 2026, arXiv:2607.22461: zbar = 0.97 -> A = 1.19 +- 0.56; zbar = 2.10 -> A = 2.86 +- 1.62;
#  all-sample 1.69 +- 0.61; joint two-bin 1.38 +- 0.53), for the committed law and the inflow branch.
# Two conventions: 'D1' (galaxy clustering amplitude b*sigma8(z) fixed by observation -> ratio of ISW sources only)
# and 'D2' (bias fixed, galaxy overdensity scales with the model's own D(z)).
# Also a dN/dz-weighted estimate per bin using broad log-normal-ish windows.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import camb
from camb.dark_energy import DarkEnergyPPF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inflow_bg import de_history, MNU

OMBH2, OMCH2, TAU, AS, H0 = 0.02237, 0.1200, 0.0544, 2.1e-9, 67.36
h = H0/100; om = (OMBH2 + OMCH2 + MNU/93.14)/h**2
zgrid = np.round(np.concatenate((np.arange(0.0, 3.01, 0.05), [3.5, 4.0])), 3)

def growth(kind, prm):
    p = camb.set_params(H0=H0, ombh2=OMBH2, omch2=OMCH2, tau=TAU, mnu=MNU, As=AS, ns=0.9649 if kind == 'lcdm' else 0.9656)
    if kind != 'lcdm':
        hist = de_history(h, om, kind, prm); de = DarkEnergyPPF(); de.set_w_a_table(hist['a'], hist['w']); p.DarkEnergy = de
    p.set_matter_power(redshifts=list(zgrid), kmax=2.0); p.NonLinear = camb.model.NonLinear_none
    r = camb.get_results(p)
    zr = np.array(r.transfer_redshifts); o = np.argsort(zr)
    s8 = np.array(r.get_sigma8())[o]; f8 = np.array(r.get_fsigma8())[o]; zr = zr[o]
    E = np.array([r.hubble_parameter(z) for z in zr])/r.hubble_parameter(0.0)
    chi = r.comoving_radial_distance(zr)
    return zr, s8, f8/s8, E, chi

cases = [('lcdm', ()), ('none', ()), ('tophat', (1.388, 0.609)), ('expdec', (3.153, 3.651))]
res = {c: growth(*c) for c in cases}
zL, DL, fL, EL, chiL = res[('lcdm', ())]
SL = EL*DL*(1-fL)
def window(z, zc, width):
    # broad quasar-like window in ln(1+z)
    x = np.log(1+z); xc = np.log(1+zc)
    w = np.exp(-0.5*((x-xc)/width)**2); return w
print('Planck params fixed; only w(z) differs. A = model/LCDM ISW-galaxy cross amplitude.')
print(f"{'case':>22s}  {'A(z=0.97) D1':>12s} {'D2':>6s}   {'A(z=2.10) D1':>12s} {'D2':>6s}   {'bin-avg 0.97':>12s} {'bin-avg 2.10':>12s}   {'A(z=0.5)':>8s} {'A(z=1.5)':>8s} {'A(z=3)':>7s}")
for c in cases:
    zr, s8, f, E, chi = res[c]
    S = E*s8*(1-f)
    r1 = S/SL                 # D1 convention
    r2 = (S*s8)/(SL*DL)       # D2 convention
    def at(z, arr): return float(np.interp(z, zr, arr))
    # bin averages (D1), weighted by window * (dchi/dz) * ISW-source-of-LCDM to mimic the template's own weighting
    out = []
    for zc, wd in [(0.97, 0.35), (2.10, 0.30)]:
        w = window(zr, zc, wd)*SL*DL; out.append(float(np.sum(w*r1)/np.sum(w)))
    label = c[0] if not c[1] else f"{c[0]}{tuple(round(v,3) for v in c[1])}"
    print(f"{label:>22s}  {at(0.97, r1):12.3f} {at(0.97, r2):6.3f}   {at(2.10, r1):12.3f} {at(2.10, r2):6.3f}   {out[0]:12.3f} {out[1]:12.3f}   {at(0.5, r1):8.3f} {at(1.5, r1):8.3f} {at(3.0, r1):7.3f}")
print('\nQuaia x Planck (2026): A(0.97) = 1.19 +- 0.56; A(2.10) = 2.86 +- 1.62; all = 1.69 +- 0.61; joint = 1.38 +- 0.53')
for c in cases:
    zr, s8, f, E, chi = res[c]; S = E*s8*(1-f); r1 = S/SL
    a1, a2 = float(np.interp(0.97, zr, r1)), float(np.interp(2.10, zr, r1))
    chi2 = ((1.19-a1)/0.56)**2 + ((2.86-a2)/1.62)**2
    label = c[0] if not c[1] else f"{c[0]}{tuple(round(v,3) for v in c[1])}"
    print(f"   {label:>22s}: pulls {(1.19-a1)/0.56:+.2f}s, {(2.86-a2)/1.62:+.2f}s ; chi2(2 bins) = {chi2:.2f}; all-sample pull vs mean of bins {(1.69-0.5*(a1+a2))/0.61:+.2f}s")
