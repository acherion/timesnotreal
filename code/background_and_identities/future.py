# Far-future entailment of w(z) = -1 + 0.133/E(z).
# (1+w) = C/E  =>  dln(rho_DE)/dt = -3H(1+w) = -3*C*H0 = constant
# => rho_DE decays EXPONENTIALLY in cosmic time. Integrate forward to see
# Omega_DE(t), the H*t trajectory, and matter re-domination.
import math

PHI = (1 + 5**0.5) / 2
C = 0.133
Om0, OD0 = 0.312, 0.688
H0 = 0.0689           # 1/Gyr  (67.4 km/s/Mpc)
t0 = 0.939 / H0       # model's H0*t0 = 0.939  -> t0 = 13.63 Gyr

# analytic decay constant
rate = 3 * C * H0     # 1/Gyr
tau = 1 / rate
print(f'DE decay: rho_DE(t) = rho_DE(t0) * exp(-(t-t0)/tau)')
print(f'  tau = 1/(3*0.133*H0) = {tau:.1f} Gyr   half-life = {tau*math.log(2):.1f} Gyr')
print(f'  cross-check vs recursion form n0*|ln(1-p)|/t0: '
      f'{(2*PHI**2)*abs(math.log(1-2/PHI**7))/13.8:.4f} vs 3C*H0 = {rate:.4f}  (1/Gyr)')
print()

# integrate forward: state = (a, rho_DE) with rho_m = Om0/a^3 (units of rho_crit,0)
def deriv(a, rD):
    E = math.sqrt(Om0 / a**3 + rD)
    da = H0 * E * a
    drD = -3 * C * H0 * rD
    return da, drD

a, rD = 1.0, OD0
t = t0
dt = 0.005
peak_OD = (0, 0.0)
peak_Ht = (0, 0.0)
recross = None
marks = {20: None, 50: None, 100: None, 200: None, 400: None, 700: None, 1000: None, 2000: None}
while t < 2005:
    # RK4
    k1 = deriv(a, rD)
    k2 = deriv(a + 0.5*dt*k1[0], rD + 0.5*dt*k1[1])
    k3 = deriv(a + 0.5*dt*k2[0], rD + 0.5*dt*k2[1])
    k4 = deriv(a + dt*k3[0], rD + dt*k3[1])
    a += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    rD += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    t += dt
    rm = Om0 / a**3
    E = math.sqrt(rm + rD)
    OD = rD / (rm + rD)
    Ht = H0 * E * t
    if OD > peak_OD[1]: peak_OD = (t, OD)
    if Ht > peak_Ht[1]: peak_Ht = (t, Ht)
    if recross is None and OD < 0.5 and t > t0: recross = t
    for m in marks:
        if marks[m] is None and t >= m:
            w = -1 + C / E
            marks[m] = (a, OD, Ht, w)

print(f'Omega_DE today: {OD0}  ->  peak Omega_DE = {peak_OD[1]:.4f} at t = {peak_OD[0]:.1f} Gyr')
print(f'H*t today: 0.939      ->  peak H*t = {peak_Ht[1]:.4f} at t = {peak_Ht[0]:.1f} Gyr')
print(f'   compare 4/phi^3 = {4/PHI**3:.4f}   (age identity requires H*t = 4/phi^3)')
print(f'matter re-dominates (Omega_DE < 0.5) at t = ' + (f'{recross:.1f} Gyr' if recross else 'beyond integration'))
print()
print('  t(Gyr)    a(t)     Omega_DE     H*t      w(z)')
for m in sorted(marks):
    a_, OD_, Ht_, w_ = marks[m]
    print(f'  {m:5d}   {a_:7.2f}   {OD_:8.4f}   {Ht_:6.3f}   {w_:+.3f}')
print()
print('late-time check: Einstein-de Sitter has H*t = 2/3 = 0.667, w_eff -> 0')
