import numpy as np
from scipy.optimize import minimize
import importlib.util, sys
spec = importlib.util.spec_from_file_location('coupled', r'coupled.py')
# re-use functions without re-running the module's fits: exec only the definitions
src = open(r'coupled.py', encoding='utf-8').read()
defs = src.split("print('physical CMB compression")[0]
ns = {}
exec(defs, ns)
background, chi2_all = ns['background'], ns['chi2_all']

print('forced transfer fraction f, with (omega_b, h) re-optimized each time:')
for f in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
    def obj(p):
        omb, h = p
        if not (0.019 < omb < 0.026 and 0.6 < h < 0.75): return 1e10
        return chi2_all(background(omb, h, f), omb)
    r = minimize(obj, [0.02267, 0.67], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-5, maxiter=2000))
    omb, h = r.x
    bg = background(omb, h, f)
    parts = chi2_all(bg, omb, parts=True)
    print(f'  f = {f:4.2f}: chi2 = {r.fun:6.2f} [CMB {parts[0]:5.2f} BAO {parts[1]:5.2f} SN {parts[2]:5.2f}]  '
          f'H0 = {100*h:.2f}  omm_early = {bg["omm_early"]:.4f}  100theta* = {bg["theta"]:.5f}')
