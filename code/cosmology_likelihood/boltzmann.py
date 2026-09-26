# Full Boltzmann-code run of the recursion model vs LCDM (CAMB).
# Outputs: R shift parameter, 100*theta_star, sigma8/S8, low-ell TT suppression from k_min,
# and the H0 that the CMB acoustic scale prefers under the model w(z).
import numpy as np
import camb
from camb.dark_energy import DarkEnergyPPF

C_DRAIN = 0.133
KMIN = 2.3e-4      # Mpc^-1 (parent r_S cutoff)
NS_MODEL, NS_LCDM = 0.9656, 0.9649
AS, K0 = 2.1e-9, 0.05
OMBH2, OMCH2, TAU, MNU = 0.02237, 0.1200, 0.0544, 0.06

def model_w_a(h):
    """Self-consistent w(a) for w = -1 + C/E(z), iterated to convergence."""
    om = (OMBH2 + OMCH2 + MNU/93.14) / h**2
    orad = 4.15e-5 / h**2
    ode0 = 1 - om - orad
    z = np.logspace(-4, 4, 3000)
    zg = np.concatenate(([0.0], z))
    E = np.sqrt(om*(1+zg)**3 + orad*(1+zg)**4 + ode0)   # start from LCDM
    for _ in range(8):
        w = -1 + C_DRAIN/E
        integ = 3*(1+w)/(1+zg)
        I = np.concatenate(([0.0], np.cumsum(0.5*(integ[1:]+integ[:-1])*np.diff(zg))))
        f = np.exp(I)                                    # rho_DE(z)/rho_DE(0)
        E = np.sqrt(om*(1+zg)**3 + orad*(1+zg)**4 + ode0*f)
    a = 1/(1+zg)
    return a[::-1], (-1 + C_DRAIN/E)[::-1]               # ascending a

def make_pars(h0, model=False, cutoff=False, linear=False):
    p = camb.set_params(H0=h0, ombh2=OMBH2, omch2=OMCH2, tau=TAU, mnu=MNU,
                        As=AS, ns=(NS_MODEL if model else NS_LCDM), lmax=2600)
    if model:
        a, w = model_w_a(h0/100)
        de = DarkEnergyPPF()
        de.set_w_a_table(a, w)
        p.DarkEnergy = de
    if cutoff:
        k = np.logspace(-5, 1.3, 2000)
        pk = AS*(k/K0)**(NS_MODEL-1) * np.exp(-(KMIN/k)**4)   # sharp-ish IR cutoff
        p.set_initial_power_table(k, pk, effective_ns_for_nonlinear=NS_MODEL)
    if linear:
        p.NonLinear = camb.model.NonLinear_none
    p.set_matter_power(redshifts=[0.0], kmax=2.0)
    return p

def observables(p):
    r = camb.get_results(p)
    d = r.get_derived_params()
    zstar = d['zstar']
    chi = r.comoving_radial_distance(zstar)              # Mpc (flat)
    h = p.H0/100
    om = (OMBH2 + OMCH2 + MNU/93.14)/h**2
    R = np.sqrt(om) * (p.H0/299792.458) * chi
    s8 = r.get_sigma8_0()
    S8 = s8*np.sqrt(om/0.3)
    cl = r.get_cmb_power_spectra(p, CMB_unit='muK')['total'][:, 0]  # lensed TT D_l
    return dict(R=R, theta=d['thetastar'], s8=s8, S8=S8, cl=cl, zstar=zstar)

print('run 1: LCDM baseline (Planck 2018 params, H0=67.36)')
lcdm = observables(make_pars(67.36))
print(f"   R = {lcdm['R']:.4f}   100theta* = {lcdm['theta']:.5f}   sigma8 = {lcdm['s8']:.4f}   S8 = {lcdm['S8']:.4f}")

print('run 2: model w(z), same H0, no cutoff')
m1 = observables(make_pars(67.36, model=True))
print(f"   R = {m1['R']:.4f}   100theta* = {m1['theta']:.5f}   sigma8 = {m1['s8']:.4f}   S8 = {m1['S8']:.4f}")

print('run 3: model w(z) + k_min cutoff + ns=0.9656 (linear transfer, cutoff vs no-cutoff)')
m2a = observables(make_pars(67.36, model=True, linear=True))
m2 = observables(make_pars(67.36, model=True, cutoff=True, linear=True))
print(f"   R = {m2['R']:.4f}   100theta* = {m2['theta']:.5f}   sigma8 = {m2['s8']:.4f}   S8 = {m2['S8']:.4f}")
print('   low-ell TT suppression from cutoff: ', end='')
for ell in (2, 3, 4, 5, 8, 12, 20, 30):
    print(f"l={ell}: {100*(m2['cl'][ell]/m2a['cl'][ell]-1):+.0f}%", end='  ')
print()

# run 4: tune H0 so the model matches LCDM's acoustic scale theta*
print('run 4: H0 tuned to match LCDM 100theta* under model w(z)')
target = lcdm['theta']
lo, hi = 63.0, 70.0
for _ in range(18):
    mid = 0.5*(lo+hi)
    t = observables(make_pars(mid, model=True))['theta']
    if t > target: hi = mid      # larger H0 -> larger theta*
    else: lo = mid
h0fit = 0.5*(lo+hi)
mfit = observables(make_pars(h0fit, model=True))
print(f"   H0(theta*-matched) = {h0fit:.2f}   R = {mfit['R']:.4f}   sigma8 = {mfit['s8']:.4f}   S8 = {mfit['S8']:.4f}")

print()
print(f"Planck measured R = 1.7502 +- 0.0046")
for name, run in (('LCDM', lcdm), ('model sameH0', m1), ('model theta*-matched', mfit)):
    print(f"   {name:22s} R = {run['R']:.4f}  ->  {(run['R']-1.7502)/0.0046:+.1f} sigma")
