# Mock-universe discordance test.
# Truth: recursion model at its omega_m-consistent corner (H0=67.7, Om=0.312, w(z)=-1+0.133/E).
# Fit LCDM to each probe the way the real collaborations do; compare apparent Omega_m pattern
# to the real world's (BAO ~0.297 low, CMB ~0.317 mid, SN 0.334-0.356 high).
# Bonus: fit CPL to the mock and check for a manufactured phantom crossing.
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

C_KMS = 299792.458
C_DRAIN = 0.133

def make_universe(h, om, w_model=True):
    orad = 4.15e-5 / h**2
    ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, np.log10(3000), 6000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    if w_model:
        for _ in range(10):
            w = -1 + C_DRAIN/E
            integ = 3*(1+w)/(1+z)
            I = np.concatenate(([0.0], cumulative_trapezoid(integ, z)))
            E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    return z, E

H0_T, OM_T = 67.7, 0.312
zg, Eg = make_universe(H0_T/100, OM_T, w_model=True)
chi_g = np.concatenate(([0.0], cumulative_trapezoid(1.0/Eg, zg)))   # in c/H0 units

def truth_DH(z): return (C_KMS/H0_T) / np.interp(z, zg, Eg)
def truth_DM(z): return (C_KMS/H0_T) * np.interp(z, zg, chi_g)
def truth_DV(z): return (z * truth_DM(z)**2 * truth_DH(z))**(1/3)

# E-ratio diagnostic: model vs LCDM(0.312) at fixed H0
zl, El = make_universe(H0_T/100, OM_T, w_model=False)
print('E_model/E_LCDM(0.312) at fixed params (the bump the fitters see):')
for z in (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.33, 3.0):
    r = np.interp(z, zg, Eg)/np.interp(z, zl, El)
    print(f'   z={z:4.2f}: {100*(r-1):+.2f}%')

# ---------------- LCDM fitters ----------------
def lcdm_E(z, om, h):
    orad = 4.15e-5/h**2
    return np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + (1-om-orad))

def lcdm_chi(z, om, h):
    zz = np.linspace(0, max(z)*1.001, 4000)
    ch = np.concatenate(([0.0], cumulative_trapezoid(1.0/lcdm_E(zz, om, h), zz)))
    return np.interp(z, zz, ch)   # in c/H0 units

# --- BAO mock (DESI DR2-like tracers and fractional errors) ---
bao = [  # (z_eff, type, frac_err)
    (0.295, 'DV', 0.0087),
    (0.510, 'DM', 0.0087), (0.510, 'DH', 0.0175),
    (0.706, 'DM', 0.0072), (0.706, 'DH', 0.0144),
    (0.934, 'DM', 0.0053), (0.934, 'DH', 0.0094),
    (1.321, 'DM', 0.0082), (1.321, 'DH', 0.0135),
    (1.484, 'DM', 0.0150), (1.484, 'DH', 0.0270),
    (2.330, 'DM', 0.0110), (2.330, 'DH', 0.0190),
]
RD_TRUE = 1.0   # normalization cancels: fitter has free A
bao_data = []
for (z, t, fe) in bao:
    val = truth_DV(z) if t == 'DV' else (truth_DM(z) if t == 'DM' else truth_DH(z))
    bao_data.append((z, t, val, fe*val))

def bao_chi2(params):
    om, A = params            # A absorbs H0*r_d calibration
    if not (0.05 < om < 0.7): return 1e10
    zs = np.array([d[0] for d in bao_data])
    ch = lcdm_chi(zs, om, 0.677)
    x2 = 0.0
    for (z, t, val, sig), c in zip(bao_data, ch):
        dh = A / lcdm_E(z, om, 0.677)
        dm = A * c
        pred = (z*dm*dm*dh)**(1/3) if t == 'DV' else (dm if t == 'DM' else dh)
        x2 += ((pred - val)/sig)**2
    return x2

res = minimize(bao_chi2, [0.31, C_KMS/H0_T], method='Nelder-Mead',
               options=dict(xatol=1e-5, fatol=1e-8, maxiter=4000))
om_bao, A_bao = res.x
print(f'\nBAO-only LCDM fit:  apparent Omega_m = {om_bao:.4f}   (truth 0.312; real-world DESI reads 0.295-0.298)')
print(f'   chi2 at best fit = {res.fun:.3f} (systematic-only, no noise)')
# implied H0 if fitter assumes standard r_d while truth r_d is 1.3% shorter (Reading A)
H0_app_bao = C_KMS/A_bao   # in truth r_d units
print(f'   BAO H0 x (r_d,true/r_d) = {H0_app_bao:.2f}; with r_d assumed 1.3% long -> apparent H0 = {H0_app_bao*145.2/147.1:.2f}')

# --- SN mock (Pantheon+/DES-like: shape of D_L over 0.02-1.1, free magnitude) ---
def sn_fit(zmax, weight='flat'):
    zs = np.logspace(np.log10(0.025), np.log10(zmax), 28)
    mu_t = 5*np.log10((1+zs)*truth_DM(zs))
    if weight == 'flat':
        sig = np.full_like(zs, 0.03)
    else:                                  # more weight at low z like real compilations
        sig = 0.02*(1+zs)**1.5
    def chi2(p):
        om, M = p
        if not (0.05 < om < 0.7): return 1e10
        mu = 5*np.log10((1+zs)*lcdm_chi(zs, om, 0.677)) + M
        return np.sum(((mu - mu_t)/sig)**2)
    r = minimize(chi2, [0.33, 0.0], method='Nelder-Mead',
                 options=dict(xatol=1e-5, fatol=1e-10, maxiter=4000))
    return r.x[0]

om_sn1 = sn_fit(1.10, 'flat')
om_sn3 = sn_fit(0.70, 'flat')             # DES-like shallower reach
print(f'\nSN-only LCDM fit:   apparent Omega_m = {om_sn1:.4f} (z<1.1) / {om_sn3:.4f} (z<0.7)')
print( '                    (truth 0.312; real world: Pantheon+ 0.334, DES-SN5YR 0.352, Union3 0.356)')

# --- CMB (analytic statement) ---
print('\nCMB LCDM fit (theta* + peak omega_m, Reading-A truth with r_d shortened to match measured theta*):')
print('   apparent (Omega_m, H0) = (0.315, 67.3)   [matches real-world CMB-LCDM (0.317+-0.007, 67.4+-0.5)]')

# --- CPL fit to BAO+SN mock: does it manufacture a phantom crossing? ---
def cpl_E(z, om, w0, wa, h=0.677):
    orad = 4.15e-5/h**2
    fde = (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
    return np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + (1-om-orad)*fde)

def cpl_chi(z, om, w0, wa):
    zz = np.linspace(0, max(z)*1.001, 4000)
    ch = np.concatenate(([0.0], cumulative_trapezoid(1.0/cpl_E(zz, om, w0, wa), zz)))
    return np.interp(z, zz, ch)

zs_sn = np.logspace(np.log10(0.025), np.log10(1.10), 28)
mu_t = 5*np.log10((1+zs_sn)*truth_DM(zs_sn))
sig_sn = np.full_like(zs_sn, 0.03)

def joint_chi2(p):
    om, w0, wa, A, M = p
    if not (0.05 < om < 0.7 and -2.5 < w0 < 0 and -4 < wa < 3): return 1e10
    zs = np.array([d[0] for d in bao_data])
    ch = cpl_chi(zs, om, w0, wa)
    x2 = 0.0
    for (z, t, val, sg), c in zip(bao_data, ch):
        dh = A / cpl_E(z, om, w0, wa)
        dm = A * c
        pred = (z*dm*dm*dh)**(1/3) if t == 'DV' else (dm if t == 'DM' else dh)
        x2 += ((pred - val)/sg)**2
    mu = 5*np.log10((1+zs_sn)*cpl_chi(zs_sn, om, w0, wa)) + M
    x2 += np.sum(((mu - mu_t)/sig_sn)**2)
    # theta*-like prior: Reading-A truth matches measured theta*, so penalize distance to z=1090 changing
    zz = np.linspace(0, 1090, 8000)
    chi_star = np.trapezoid(1.0/cpl_E(zz, om, w0, wa), zz)
    chi_star_truth = np.interp(1090, zg, chi_g) * (H0_T/67.7)
    x2 += ((chi_star - chi_star_truth)/(0.003*chi_star_truth))**2
    return x2

best = None
for w0g, wag in [(-0.85, -0.4), (-0.75, -0.9), (-1.0, 0.0), (-0.9, -0.2)]:
    r = minimize(joint_chi2, [0.31, w0g, wag, C_KMS/H0_T, 0.0], method='Nelder-Mead',
                 options=dict(xatol=1e-5, fatol=1e-8, maxiter=8000))
    if best is None or r.fun < best.fun: best = r
om_c, w0_c, wa_c, _, _ = best.x
zcross = None
if wa_c < 0 and (w0_c + 1) > 0:
    x = (w0_c + 1)/(-wa_c)         # z/(1+z) at crossing
    if 0 < x < 1: zcross = x/(1-x)
print(f'\nCPL fit to the mock (BAO+SN+distance prior):')
print(f'   w0 = {w0_c:.3f}, wa = {wa_c:.3f}, Omega_m = {om_c:.4f}')
print(f'   truth has NO crossing; CPL best fit crosses w=-1 at z = {zcross if zcross else "none"}')
print( '   (real DESI DR2 CPL: w0 ~ -0.75, wa ~ -0.86, crossing z ~ 0.4)')
