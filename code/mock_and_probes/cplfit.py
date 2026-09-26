# CPL fit to the model-universe mock, done properly:
# - A (BAO norm) and M (SN norm) profiled out analytically
# - Reading-A distance prior: CPL fitter assumes standard r_d, so it must reach
#   chi*(1090) = 1.0131 x truth (the 1.3% r_d bookkeeping gap) while fitting the
#   same BAO+SN shapes. Question: does the best fit manufacture a phantom crossing?
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

C_KMS = 299792.458
C_DRAIN = 0.133
H0_T, OM_T = 67.7, 0.312

def make_universe(h, om):
    orad = 4.15e-5/h**2
    ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, np.log10(3000), 6000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    for _ in range(10):
        w = -1 + C_DRAIN/E
        I = np.concatenate(([0.0], cumulative_trapezoid(3*(1+w)/(1+z), z)))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    return z, E

zg, Eg = make_universe(H0_T/100, OM_T)
chi_g = np.concatenate(([0.0], cumulative_trapezoid(1.0/Eg, zg)))
def tDH(z): return (C_KMS/H0_T)/np.interp(z, zg, Eg)
def tDM(z): return (C_KMS/H0_T)*np.interp(z, zg, chi_g)
def tDV(z): return (z*tDM(z)**2*tDH(z))**(1/3)
CHI_STAR_T = (C_KMS/H0_T)*np.interp(1090.0, zg, chi_g)
CHI_TARGET = CHI_STAR_T * 1.0131          # standard-r_d bookkeeping under Reading A

bao = [(0.295,'DV',0.0087),(0.510,'DM',0.0087),(0.510,'DH',0.0175),
       (0.706,'DM',0.0072),(0.706,'DH',0.0144),(0.934,'DM',0.0053),(0.934,'DH',0.0094),
       (1.321,'DM',0.0082),(1.321,'DH',0.0135),(1.484,'DM',0.0150),(1.484,'DH',0.0270),
       (2.330,'DM',0.0110),(2.330,'DH',0.0190)]
bao_val = np.array([tDV(z) if t=='DV' else (tDM(z) if t=='DM' else tDH(z)) for z,t,_ in bao])
bao_sig = np.array([fe for _,_,fe in bao]) * bao_val

zs_sn = np.logspace(np.log10(0.025), np.log10(1.10), 28)
mu_t = 5*np.log10((1+zs_sn)*tDM(zs_sn))
sig_sn = 0.03

def cpl_model(om, w0, wa):
    h = 0.677
    orad = 4.15e-5/h**2
    z = np.concatenate(([0.0], np.logspace(-4, np.log10(1200), 4000)))
    fde = (1+z)**(3*(1+w0+wa))*np.exp(-3*wa*z/(1+z))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + (1-om-orad)*fde)
    chi = np.concatenate(([0.0], cumulative_trapezoid(1.0/E, z)))
    return z, E, chi

def chi2(p):
    om, w0, wa = p
    if not (0.1 < om < 0.6 and -2 < w0 < -0.2 and -3 < wa < 2): return 1e10
    z, E, chi = cpl_model(om, w0, wa)
    # BAO shapes (unit-normalized; A profiled)
    shp = []
    for (zz, t, _), in zip(bao,):
        dm = np.interp(zz, z, chi); dh = 1/np.interp(zz, z, E)
        shp.append((zz*dm*dm*dh)**(1/3) if t=='DV' else (dm if t=='DM' else dh))
    shp = np.array(shp)
    A = np.sum(bao_val*shp/bao_sig**2)/np.sum(shp*shp/bao_sig**2)
    x2 = np.sum(((A*shp - bao_val)/bao_sig)**2)
    # SN (M profiled)
    mu = 5*np.log10((1+zs_sn)*np.interp(zs_sn, z, chi))
    dmu = mu_t - mu
    M = np.mean(dmu)
    x2 += np.sum(((dmu - M)/sig_sn)**2)
    # CMB distance prior (Reading A bookkeeping)
    chi_star = (C_KMS/67.7)*np.interp(1090.0, z, chi)   # overall c/H0 scale cancels against A-like freedom...
    # the fitter's absolute distance scale: use H0 as free via effective rescale absorbed in prior width
    x2 += ((chi_star - CHI_TARGET)/(0.003*CHI_TARGET))**2
    return x2

best = None
for w0g in (-0.7, -0.85, -1.0, -1.15):
    for wag in (-1.2, -0.6, 0.0, 0.6):
        r = minimize(chi2, [0.315, w0g, wag], method='Nelder-Mead',
                     options=dict(xatol=1e-5, fatol=1e-9, maxiter=6000))
        if best is None or r.fun < best.fun: best = r
om_c, w0_c, wa_c = best.x
print(f'CPL best fit to model-universe mock: Omega_m = {om_c:.4f}, w0 = {w0_c:.3f}, wa = {wa_c:.3f}, chi2 = {best.fun:.2f}')
zc = None
if wa_c < 0 and (1+w0_c) > 0:
    x = (1+w0_c)/(-wa_c)
    if 0 < x < 1: zc = x/(1-x)
print(f'crossing of w = -1: z = {zc:.2f}' if zc else 'no crossing in 0<z<inf')
print(f'(real DESI DR2 CPL: Omega_m ~ 0.319, w0 ~ -0.75, wa ~ -0.86, crossing z ~ 0.4)')
# also report w at pivot for sanity
if zc is not None or True:
    zp = 0.34
    wp = w0_c + wa_c*zp/(1+zp)
    print(f'w at pivot z=0.34: {wp:.3f}   (truth: -0.892)')
