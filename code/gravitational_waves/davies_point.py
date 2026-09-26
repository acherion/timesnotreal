# Verify the claimed golden ratio in Kerr black hole thermodynamics (Davies point).
# Kerr, G=c=hbar=k=1:  a = J/M, r+ = M + sqrt(M^2-a^2)
#   T = (r+ - r-)/(4 pi (r+^2 + a^2)),  S = pi (r+^2 + a^2)
# Heat capacity at fixed J: C_J = T (dS/dT)_J -> diverges where (dT/dM)_J = 0.
# Claim (QGR video): the sign change happens at M^4/J^2 = phi, i.e. (a/M)^2 = 1/phi.
# Competing memory: Davies point at (a/M)^2 = 2 sqrt(3) - 3.
import numpy as np
from scipy.optimize import brentq
PHI = (1 + 5**0.5)/2

J = 1.0
def T_of_M(M):
    a = J/M
    if M*M < a*a: return np.nan
    rp = M + np.sqrt(M*M - a*a)
    rm = M - np.sqrt(M*M - a*a)
    return (rp - rm)/(4*np.pi*(rp*rp + a*a))

def dT_dM(M, h=1e-7):
    return (T_of_M(M+h) - T_of_M(M-h))/(2*h)

# extremal at M^2 = a^2 = J/... M^4 = J^2 -> M = 1. Search M in (1, 3)
Mc = brentq(dT_dM, 1.0001, 3.0)
ac = J/Mc
print('Davies point (C_J divergence):')
print('  M^4/J^2      =', f'{Mc**4:.10f}')
print('  phi          =', f'{PHI:.10f}')
print('  (a/M)^2      =', f'{(ac/Mc)**2:.10f}')
print('  1/phi        =', f'{1/PHI:.10f}')
print('  2sqrt(3)-3   =', f'{2*np.sqrt(3)-3:.10f}')
print('  a/M          =', f'{ac/Mc:.10f}')
print('  1/sqrt(phi)  =', f'{PHI**-0.5:.10f}')
print('  model universal remnant spin = 0.6865')

# sign check on either side of Mc: C_J = T dS/dT
def S_of_M(M):
    a = J/M
    rp = M + np.sqrt(M*M - a*a)
    return np.pi*(rp*rp + a*a)
def C_J(M, h=1e-6):
    dS = (S_of_M(M+h) - S_of_M(M-h))/(2*h)
    dT = dT_dM(M, h)
    return T_of_M(M)*dS/dT
for M in [Mc*0.98, Mc*1.02]:
    print(f'  C_J at M = {M/Mc:.2f} Mc: {C_J(M):+.3f}')
