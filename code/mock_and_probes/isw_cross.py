# ISW-galaxy cross-correlation amplitude: model vs LCDM.
# Source: dPhi/deta ∝ Om H0^2 * H(z) * D(z) * (f(z)-1); galaxy overdensity ∝ D(z).
# Ratio of cross-spectrum integrands at the sample redshift: A(z) = [E D^2 (f-1)]_model / [E D^2 (f-1)]_LCDM
# (same Om, H0, A_s -> only w(z) differs). Also a rough LRG-like sample average.
import numpy as np
import camb
from camb.dark_energy import DarkEnergyPPF

C_DRAIN, OMBH2, OMCH2, TAU, MNU, AS = 0.133, 0.02237, 0.1200, 0.0544, 0.06, 2.1e-9
zgrid = np.round(np.concatenate((np.arange(0.0, 2.01, 0.05), [2.5, 3.0])), 3)

def model_w_a(h, om):
    orad = 4.15e-5/h**2; ode0 = 1 - om - orad
    z = np.concatenate(([0.0], np.logspace(-4, 4, 3000)))
    E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0)
    for _ in range(8):
        w = -1 + C_DRAIN/E
        I = np.concatenate(([0.0], np.cumsum(0.5*(3*(1+w[1:])/(1+z[1:]) + 3*(1+w[:-1])/(1+z[:-1]))*np.diff(z))))
        E = np.sqrt(om*(1+z)**3 + orad*(1+z)**4 + ode0*np.exp(I))
    return (1/(1+z))[::-1], (-1 + C_DRAIN/E)[::-1]

def growth(model):
    p = camb.set_params(H0=67.36, ombh2=OMBH2, omch2=OMCH2, tau=TAU, mnu=MNU, As=AS, ns=0.9656 if model else 0.9649)
    h = 0.6736; om = (OMBH2 + OMCH2 + MNU/93.14)/h**2
    if model:
        a, w = model_w_a(h, om); de = DarkEnergyPPF(); de.set_w_a_table(a, w); p.DarkEnergy = de
    p.set_matter_power(redshifts=list(zgrid), kmax=2.0); p.NonLinear = camb.model.NonLinear_none
    r = camb.get_results(p)
    zr = np.array(r.transfer_redshifts); o = np.argsort(zr)
    s8 = np.array(r.get_sigma8())[o]; f8 = np.array(r.get_fsigma8())[o]; zr = zr[o]
    E = np.array([r.hubble_parameter(z) for z in zr])/r.hubble_parameter(0.0)
    f = f8/s8
    return zr, s8, f, E

zL, DL, fL, EL = growth(False)
zM, DM, fM, EM = growth(True)
SL = EL*DL*(fL-1); SM = EM*DM*(fM-1)           # ISW source (sign negative: potentials decay)
ratio_cross = (SM*DM)/(SL*DL)                  # cross-correlation integrand ratio
ratio_auto = (SM/SL)**2                        # ISW auto-power ratio

print('ISW source and cross-correlation amplitude ratio, model/LCDM (Planck params, only w(z) differs):')
print('   z     f_LCDM  f_model   (1-f)_M/(1-f)_L   A_cross   A_auto')
for z in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
    i = np.argmin(abs(zL - z))
    print(f'  {z:4.2f}   {fL[i]:.4f}  {fM[i]:.4f}      {(1-fM[i])/(1-fL[i]):.3f}          {ratio_cross[i]:.3f}     {ratio_auto[i]:.3f}')
# LRG-like sample (z ~ 0.3-0.8, flat weight) and a DESI-LRG-ish window
sel = (zL >= 0.3) & (zL <= 0.8)
print(f'\nLRG-like sample (0.3<z<0.8) mean cross amplitude ratio: {np.mean(ratio_cross[sel]):.3f}')
sel2 = (zL >= 0.6) & (zL <= 1.5)
print(f'ELG/QSO-like sample (0.6<z<1.5) mean cross amplitude ratio: {np.mean(ratio_cross[sel2]):.3f}')
print(f'\nsigma8(0): LCDM {DL[0]:.4f}, model {DM[0]:.4f};  f(0): LCDM {fL[0]:.4f}, model {fM[0]:.4f}')
