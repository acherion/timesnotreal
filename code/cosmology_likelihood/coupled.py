# Partial DE->DM transfer ("fuel processed into structure", taken literally at fraction f of the drain).
# Joint fit: physical CMB compression {100theta*, omega_m(early), omega_b} + DESI DR2 BAO + SN shape (0.330+-0.015).
# Cases: LCDM; locked model f=0; locked model + free f in [0,1].
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize

C_KMS = 299792.458
C0 = 0.133
OM0 = 0.312
OMNU_H2 = 0.06/93.14
ZSTAR = 1089.9
RS_OVER_RD = 1/1.0184           # r_*/r_d (Planck 2018: 144.43/147.09)

# ---- data ----
THETA_MEAN, THETA_SIG = 1.04109, 0.00030
OMM_MEAN, OMM_SIG = 0.1430, 0.0011          # early-universe (recombination) matter density from peak morphology
OMB_MEAN, OMB_SIG = 0.02237, 0.00015
BAO_DV = (0.295, 7.944, 0.075)
BAO = [(0.510, 13.588, 0.167, 21.863, 0.427, -0.459), (0.706, 17.351, 0.177, 19.455, 0.330, -0.404),
       (0.934, 21.576, 0.152, 17.641, 0.193, -0.416), (1.321, 27.601, 0.318, 14.176, 0.221, -0.434),
       (1.484, 30.512, 0.760, 12.817, 0.516, -0.500), (2.330, 38.988, 0.531,  8.632, 0.101, -0.431)]
ZS_SN = np.logspace(np.log10(0.025), np.log10(1.10), 30)
OM_SN, SIG_OM_SN = 0.330, 0.015

def rd_aubourg(omcb, omb, omnu=OMNU_H2):
    return 55.154*np.exp(-72.3*(omnu + 0.0006)**2)/(omcb**0.25351 * omb**0.12807)

def background(omb, h, f, om0=OM0, lcdm=False, omc_lcdm=None):
    orad = 4.15e-5/h**2
    if lcdm:
        om0 = (omb + omc_lcdm + OMNU_H2)/h**2
    ode0 = 1 - om0 - orad
    def rhs(lna, y):
        rm, rd = y
        a = np.exp(lna)
        E = np.sqrt(rm + rd + orad*a**-4)
        drain = 3*C0*rd/E if not lcdm else 0.0     # d rho_DE / d ln a
        return [-3*rm + f*drain, -drain]
    sol = solve_ivp(rhs, [0.0, -np.log(1+3000.0)], [om0, ode0], dense_output=True, rtol=1e-9, atol=1e-13)
    def E_of_z(z):
        z = np.atleast_1d(z)
        out = []
        for zz in z:
            rm, rd = sol.sol(-np.log(1+zz))
            out.append(np.sqrt(rm + rd + orad*(1+zz)**4))
        return np.array(out)
    # early matter density (comoving, in rho_crit0 units) at recombination
    rm_early = sol.sol(-np.log(1+1100.0))[0]/(1+1100.0)**3
    omm_early = rm_early*h*h
    # distances
    zg = np.concatenate((np.linspace(0, 3, 1500), np.logspace(np.log10(3.01), np.log10(ZSTAR), 1500)))
    Eg = E_of_z(zg)
    chi = np.concatenate(([0.0], cumulative_trapezoid(1.0/Eg, zg)))*(C_KMS/(100*h))
    DM = lambda z: np.interp(z, zg, chi)
    DH = lambda z: (C_KMS/(100*h))/E_of_z(z)
    omcb_early = omm_early - OMNU_H2
    rd = rd_aubourg(omcb_early, omb)
    rstar = rd*RS_OVER_RD
    theta100 = 100*rstar/DM(ZSTAR)
    return dict(theta=theta100, omm_early=omm_early, rd=rd, DM=DM, DH=DH, E=E_of_z, om0=om0)

def sn_shape_mu(om):
    zz = np.linspace(0, 1.2, 3000)
    E = np.sqrt(om*(1+zz)**3 + (1-om))
    ch = np.concatenate(([0.0], cumulative_trapezoid(1.0/E, zz)))
    return 5*np.log10((1+ZS_SN)*np.interp(ZS_SN, zz, ch))
MU_SN = sn_shape_mu(OM_SN)
_d = sn_shape_mu(OM_SN + 0.01) - MU_SN; _d -= _d.mean()
S_SN = np.sqrt(np.sum((_d/0.01)**2))*SIG_OM_SN

def chi2_all(bg, omb, parts=False):
    x_cmb = ((bg['theta'] - THETA_MEAN)/THETA_SIG)**2 + ((bg['omm_early'] - OMM_MEAN)/OMM_SIG)**2 + ((omb - OMB_MEAN)/OMB_SIG)**2
    rd = bg['rd']
    z0 = BAO_DV[0]
    dv = (z0*bg['DM'](z0)**2*bg['DH'](z0)[0])**(1/3)/rd
    x_bao = ((dv - BAO_DV[1])/BAO_DV[2])**2
    for (z, dmv, sm, dhv, sh, rho) in BAO:
        dvec = np.array([bg['DM'](z)/rd - dmv, bg['DH'](z)[0]/rd - dhv])
        cov = np.array([[sm*sm, rho*sm*sh], [rho*sm*sh, sh*sh]])
        x_bao += dvec @ np.linalg.inv(cov) @ dvec
    mu = 5*np.log10((1+ZS_SN)*bg['DM'](ZS_SN))
    d = MU_SN - mu; d -= d.mean()
    x_sn = np.sum((d/S_SN)**2)
    return (x_cmb, x_bao, x_sn) if parts else x_cmb + x_bao + x_sn

def fit(kind):
    if kind == 'lcdm':
        def obj(p):
            omb, omc, h = p
            if not (0.019 < omb < 0.026 and 0.08 < omc < 0.16 and 0.6 < h < 0.75): return 1e10
            return chi2_all(background(omb, h, 0.0, lcdm=True, omc_lcdm=omc), omb)
        r = minimize(obj, [0.02237, 0.1200, 0.6736], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-5, maxiter=3000))
        omb, omc, h = r.x; bg = background(omb, h, 0.0, lcdm=True, omc_lcdm=omc); f = 0.0
    elif kind == 'locked':
        def obj(p):
            omb, h = p
            if not (0.019 < omb < 0.026 and 0.6 < h < 0.75): return 1e10
            return chi2_all(background(omb, h, 0.0), omb)
        r = minimize(obj, [0.02237, 0.677], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-5, maxiter=3000))
        omb, h = r.x; bg = background(omb, h, 0.0); f = 0.0
    else:
        def obj(p):
            omb, h, f = p
            if not (0.019 < omb < 0.026 and 0.6 < h < 0.75 and 0.0 <= f <= 1.0): return 1e10
            return chi2_all(background(omb, h, f), omb)
        best = None
        for f0 in (0.05, 0.12, 0.25, 0.5):
            r = minimize(obj, [0.02237, 0.68, f0], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-5, maxiter=4000))
            if best is None or r.fun < best.fun: best = r
        r = best
        omb, h, f = r.x; bg = background(omb, h, f)
    parts = chi2_all(bg, omb, parts=True)
    print(f'{kind:8s}: chi2 = {r.fun:6.2f}  [CMB {parts[0]:5.2f}, BAO {parts[1]:5.2f}, SN {parts[2]:5.2f}]'
          f'   H0 = {100*h:.2f}  omb = {omb:.5f}  Om0 = {bg["om0"]:.4f}  omm_early = {bg["omm_early"]:.4f}'
          f'  f = {f:.3f}  100theta* = {bg["theta"]:.5f}  r_d = {bg["rd"]:.2f}')
    return r.fun, bg, f, h

print('physical CMB compression {100theta*, omega_m(early), omega_b} + DESI DR2 BAO + SN(0.330+-0.015)')
c1, bg1, _, _ = fit('lcdm')
c2, bg2, _, _ = fit('locked')
c3, bg3, f3, h3 = fit('coupled')
print(f'\ndelta-chi2 vs LCDM:  locked model {c2-c1:+.2f}   coupled model {c3-c1:+.2f}   (coupled adds 1 parameter)')
# BAO tilt signature of the coupled best fit vs the locked model
print('\ncoupled best fit vs locked model: H(z) and D_M(z) shifts at BAO redshifts')
for z in (0.3, 0.5, 0.7, 1.0, 1.5, 2.33):
    print(f'   z={z:4.2f}: dH/H = {100*(bg3["E"](z)[0]*h3/(bg2["E"](z)[0]*0.6664)-1):+.2f}%')
