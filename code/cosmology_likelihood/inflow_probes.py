import os, sys
# Growth and ISW probes for the inflow branch vs committed model vs LCDM (Planck params; only w(z) differs).
# Usage: python inflow_probes.py tophat 1.388 0.609 [expdec 3.0 3.2 ...]  (pairs of family + params, separated by family names)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import camb
from camb.dark_energy import DarkEnergyPPF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inflow_bg import de_history, MNU

OMBH2, OMCH2, TAU, AS, H0 = 0.02237, 0.1200, 0.0544, 2.1e-9, 67.36
h = H0/100; om = (OMBH2 + OMCH2 + MNU/93.14)/h**2
zgrid = np.round(np.concatenate((np.arange(0.0, 2.01, 0.05), [2.5, 3.0])), 3)

def growth(kind, prm):
    p = camb.set_params(H0=H0, ombh2=OMBH2, omch2=OMCH2, tau=TAU, mnu=MNU, As=AS, ns=0.9649 if kind == 'lcdm' else 0.9656)
    if kind != 'lcdm':
        hist = de_history(h, om, kind, prm); de = DarkEnergyPPF(); de.set_w_a_table(hist['a'], hist['w']); p.DarkEnergy = de
    p.set_matter_power(redshifts=list(zgrid), kmax=2.0); p.NonLinear = camb.model.NonLinear_none
    r = camb.get_results(p)
    zr = np.array(r.transfer_redshifts); o = np.argsort(zr)
    s8 = np.array(r.get_sigma8())[o]; f8 = np.array(r.get_fsigma8())[o]; zr = zr[o]
    E = np.array([r.hubble_parameter(z) for z in zr])/r.hubble_parameter(0.0)
    return zr, s8, f8/s8, E, f8

# parse argv: family name followed by floats
cases = [('lcdm', ()), ('none', ())]
i = 1
while i < len(sys.argv):
    kind = sys.argv[i]; j = i + 1; vals = []
    while j < len(sys.argv):
        try: vals.append(float(sys.argv[j])); j += 1
        except ValueError: break
    cases.append((kind, tuple(vals))); i = j
res = {c: growth(*c) for c in cases}
zL, DL, fL, EL, f8L = res[('lcdm', ())]
SL = EL*DL*(fL-1)
print('Planck params fixed (H0 67.36, omega_m 0.1430); only w(z) differs.')
print(f"{'case':>28s}  sigma8(0)   S8     fs8(0.4)/LCDM  fs8(0.8)/LCDM   ISW cross ratio at z = 0.3 / 0.5 / 0.8 / 1.0 / 1.5   LRG(0.3-0.8)  ELG/QSO(0.6-1.5)")
for c in cases:
    zr, s8, f, E, f8 = res[c]
    S = E*s8*(f-1); rc = (S*s8)/(SL*DL)
    def at(z, arr): return arr[np.argmin(abs(zr - z))]
    label = c[0] if not c[1] else f"{c[0]}{tuple(round(v,3) for v in c[1])}"
    sel = (zr >= 0.3) & (zr <= 0.8); sel2 = (zr >= 0.6) & (zr <= 1.5)
    print(f"{label:>28s}  {s8[0]:.4f}   {s8[0]*np.sqrt(om/0.3):.4f}   {at(0.4, f8)/at(0.4, f8L):.4f}        {at(0.8, f8)/at(0.8, f8L):.4f}        "
          f"{at(0.3, rc):.3f} / {at(0.5, rc):.3f} / {at(0.8, rc):.3f} / {at(1.0, rc):.3f} / {at(1.5, rc):.3f}      {np.mean(rc[sel]):.3f}        {np.mean(rc[sel2]):.3f}")
