# Supporting numbers for tasks 2-3 (report only, not for site)
import math

PHI = (1 + 5**0.5) / 2
G, c = 6.674e-11, 2.998e8
MPl, Msun = 2.176434e-8, 1.989e30

# 1. Kerr horizon oblateness vs spin: C_polar / C_equatorial
#    C_eq = 4*pi*M exactly (any spin). C_polar = 2 * int_0^pi sqrt(r+^2 + a^2 cos^2 th) dth  (units GM/c^2, M=1)
def oblate(a):
    rp = 1 + math.sqrt(1 - a*a)
    n = 4000
    s = 0.0
    for i in range(n):
        th = (i + 0.5) * math.pi / n
        s += math.sqrt(rp*rp + a*a*math.cos(th)**2) * (math.pi / n)
    Cp = 2 * s
    return Cp / (4 * math.pi)

print('Kerr horizon polar/equatorial circumference ratio:')
for a, label in [(0.686, 'universal merger remnant'), (0.9, 'high spin'),
                 (0.998, 'Thorne accretion limit'), (1.0, 'extremal')]:
    r = oblate(a)
    print(f'  a = {a:5.3f} ({label:26s}): C_p/C_eq = {r:.4f}  -> {100*(1-r):.1f}% shorter')

# 2. tau_ring / t_cross identity (both in GM/c^3): tau = 2Q/omega, Q = 2phi, omega = phi^3/8
tau = 2 * (2*PHI) / (PHI**3 / 8)
tcross = 8 / PHI**3
print(f'\nring damping time = {tau:.4f} GM/c^3; interior birth-completion = {tcross:.4f} GM/c^3')
print(f'ratio = {tau/tcross:.6f} vs 4*phi = {4*PHI:.6f}  (exact by algebra)')

# 3. Where a human sits on the mass ladder (rung = 3.8e11)
Mparent = 9.1e52
rung = (Mparent / MPl)**(1 / (2*PHI**2))
for m, label in [(70, 'human'), (5.97e24, 'Earth'), (Msun, 'Sun'), (7.3e22, 'Moon')]:
    depth = math.log(m / MPl) / math.log(rung)
    print(f'  {label:6s}: {depth:.2f} rungs above Planck floor (of {2*PHI**2:.3f} total)')
gm = math.sqrt(MPl * Mparent)
print(f'  log-midpoint of our level: {gm:.2e} kg (Moon = 7.3e22 kg)')

# 4. Golden-mean-shift capacity (refractory neuron = RLL(1,inf) channel)
import math
print(f'\ncapacity of no-repeat binary channel: ln(phi) = {math.log(PHI):.4f} nats = {math.log(PHI)/math.log(2):.4f} bits/slot')

# 5. n0 decompositions (curios)
print(f'\nn0 = 2phi^2 = {2*PHI**2:.6f} = phi^3 + 1 = {PHI**3 + 1:.6f} = 5 + 1/phi^3 = {5 + PHI**-3:.6f}')
