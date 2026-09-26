# Bekenstein-Mukhanov area quantization: is Delta A = 4 ln(phi) lP^2 observable in ringdowns?
# First law: dM c^2 = (kappa c^2 / 8 pi G) dA  ->  emitted quantum hbar*w = kappa*hbar*ln(k)/(2 pi c)
# Schwarzschild kappa = c^4/(4GM)  ->  line spacing f_q = ln(k) c^3/(16 pi^2 G M)
import math
phi = (1+5**0.5)/2
G, c, Msun = 6.674e-11, 2.998e8, 1.989e30

M = 62*Msun                      # GW150914-like remnant
tM = G*M/c**3                    # GM/c^3 in seconds
fq = {k: math.log(k)/(16*math.pi**2*tM) for k in [phi, 2, 3]}

# dominant QNM at chi=0.69: M*omega = 0.5267 - 0.0813i (Leaver, June 2026)
wR, wI = 0.5267, 0.0813
f_qnm = wR/(2*math.pi*tM)
tau = tM/wI
Gamma = 1/(math.pi*tau)          # linewidth of an exponentially damped line

print(f'GM/c^3 = {tM:.3e} s   f_QNM = {f_qnm:.1f} Hz   tau = {tau*1000:.2f} ms   linewidth = {Gamma:.1f} Hz')
for k, f in fq.items():
    name = 'phi' if abs(k-phi) < 1e-9 else str(k)
    print(f'  k = {name:>3}: line spacing = {f:6.2f} Hz   spacing/linewidth = {f/Gamma:.3f}')

# mass independence: both spacing and linewidth scale as 1/M
Q = wR/(2*wI)
ratio = math.log(phi)*Q/(2*math.pi*wR)   # = (fq/Gamma), algebraically mass-free
print(f'\nspacing/linewidth (any mass, k=phi) = ln(phi)*Q/(2 pi M*wR) = {ratio:.3f}')
print(f'Q needed to resolve adjacent lines (spacing > linewidth): {Q/ratio:.0f}  (actual Q = {Q:.2f})')
