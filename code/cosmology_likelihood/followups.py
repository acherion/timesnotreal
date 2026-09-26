import os
# Follow-ups at the best fits: (1) k_min cutoff test on low-ell TT; (2) A_lens scan on the full likelihood.
import sys, json, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import camb
from camb.dark_energy import DarkEnergyPPF
from cobaya.likelihoods.planck_2018_highl_plik.TTTEEE_lite_native import TTTEEE_lite_native
from cobaya.likelihoods.planck_2018_lowl.TT import TT as LowTT
from cobaya.likelihoods.planck_2018_lowl.EE import EE as LowEE
PK = os.environ.get('COBAYA_PACKAGES_PATH', 'cobaya_packages')
HI, LT, LE = TTTEEE_lite_native(packages_path=PK), LowTT(packages_path=PK), LowEE(packages_path=PK)
C_DRAIN, MNU, KMIN = 0.133, 0.06, 2.3e-4

def load(case):
    txt = open(rf'plikfit_{case}.log', encoding='utf-8').read()
    m = re.findall(r'RESULT (\{.*\})', txt)
    return json.loads(m[-1])

def model_w_a(h, om):
    orad = 4.15e-5/h**2; ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, 4, 3000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        I = np.concatenate(([0.0], np.cumsum(0.5*(3*(1+w[1:])/(1+z[1:]) + 3*(1+w[:-1])/(1+z[:-1]))*np.diff(z))))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    return (1/(1+z))[::-1], (-1 + C_DRAIN/E)[::-1]

def spectra(bf, model, cutoff=False, alens=1.0, linear=False):
    p = camb.set_params(H0=bf['H0'], ombh2=bf['ombh2'], omch2=bf['omch2'], tau=bf['tau'], mnu=MNU,
                        As=np.exp(bf['lnAs'])*1e-10, ns=bf['ns'], lmax=2700, lens_potential_accuracy=1)
    p.Alens = alens
    if linear:
        p.NonLinear = camb.model.NonLinear_none
    if model:
        h = bf['H0']/100; om = (bf['ombh2'] + bf['omch2'] + MNU/93.14)/h**2
        a, w = model_w_a(h, om); de = DarkEnergyPPF(); de.set_w_a_table(a, w); p.DarkEnergy = de
    if cutoff:
        k = np.logspace(-5, 1.3, 2000)
        pk = np.exp(bf['lnAs'])*1e-10*(k/0.05)**(bf['ns']-1)*np.exp(-(KMIN/k)**4)
        p.set_initial_power_table(k, pk, effective_ns_for_nonlinear=bf['ns'])
    r = camb.get_results(p)
    return r.get_cmb_power_spectra(p, CMB_unit='muK')['total']

def pieces(cl, Ap):
    return HI.get_chi_squared(0, cl[:, 0], cl[:, 3], cl[:, 1], Ap), -2*LT.log_likelihood(cl[:, 0], Ap), -2*LE.log_likelihood(cl[:, 1], Ap)

for case, model in (('lcdm', False), ('model', True)):
    bf = load(case)
    print(f'=== {case}: best fit -2lnL = {bf["neg2logL"]:.2f}  (H0={bf["H0"]:.2f}, Om={bf["Om"]:.4f}, ns={bf["ns"]:.4f}, S8={bf["S8"]:.3f}, 100theta*={bf["thetastar"]:.5f})')
    base = pieces(spectra(bf, model, linear=True), bf['A_planck'])
    cut = pieces(spectra(bf, model, cutoff=True, linear=True), bf['A_planck'])
    print(f'   k_min cutoff test (low-ell TT -2lnL, linear transfer both): no cutoff {base[1]:.2f}  ->  cutoff {cut[1]:.2f}   (delta {cut[1]-base[1]:+.2f})')
    print('   A_lens scan (total -2lnL relative to A_lens=1):')
    for al in (0.90, 1.00, 1.10, 1.20, 1.30):
        pc = pieces(spectra(bf, model, alens=al), bf['A_planck'])
        tot = pc[0] + pc[1] + pc[2]
        print(f'      A_lens = {al:.2f}: total {tot:9.2f}  (plik {pc[0]:8.2f})')
    print()
