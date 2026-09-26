# Brainstorm follow-up: algebraic reduction of the alpha coefficient pattern.
# Claim to test: phi^11 + phi^9 = sqrt5*phi^10 = D^2 * phi^9, where D^2 = 1+phi^2 = phi*sqrt5
# is the doubled-Fibonacci total quantum dimension (the entropy-constant object ln(phi*sqrt5)).
# So 1/alpha = (phi^11 + phi^9 - 2 phi^2)/2  ==  (D^2 * phi^(N+d) - d * phi^d)/d.
# Also: Zeckendorf/Binet decomposition, and uniqueness of the Fibonacci part.
import math
from fractions import Fraction

PHI = (1 + 5**0.5) / 2
S5 = 5**0.5

lhs = PHI**11 + PHI**9
print(f'phi^11 + phi^9      = {lhs:.10f}')
print(f'sqrt5 * phi^10      = {S5 * PHI**10:.10f}')
D2 = 1 + PHI**2
print(f'D^2 * phi^9         = {D2 * PHI**9:.10f}   (D^2 = 1+phi^2 = phi*sqrt5 = {D2:.10f})')
inv_alpha_ladder = (PHI**11 + PHI**9 - 2*PHI**2) / 2
inv_alpha_D2     = (D2 * PHI**9 - 2 * PHI**2) / 2
print(f'1/alpha (ladder)    = {inv_alpha_ladder:.6f}')
print(f'1/alpha (D^2 form)  = {inv_alpha_D2:.6f}   identical: {abs(inv_alpha_ladder-inv_alpha_D2) < 1e-12}')

# Binet integer decomposition: 2/alpha = (a + b*sqrt5)/2 with a,b integers
# phi^n = (L_n + F_n sqrt5)/2
F = [0,1]; L = [2,1]
for i in range(2, 25):
    F.append(F[-1]+F[-2]); L.append(L[-1]+L[-2])
a = L[11] + L[9] - 2*L[2]
b = F[11] + F[9] - 2*F[2]
print(f'\n2*(1/alpha) exact  = ({a} + {b}*sqrt5)/2,  a = L11+L9-2L2 = {a},  b = F11+F9-2F2 = {b}')
print(f'b = {b} = (N+2d)^2 = 11^2: {b == 11**2}')
print(f'check: F11+F9 = L10 = {F[11]+F[9]} (={L[10]}), L11+L9 = 5*F10 = {L[11]+L[9]} (={5*F[10]})')

# Uniqueness probe: integer combos c1*F11 + c2*F9 + c3*F2 = 121 with |c| <= 3
sols = []
for c1 in range(-3,4):
    for c2 in range(-3,4):
        for c3 in range(-3,4):
            if c1*F[11] + c2*F[9] + c3*F[2] == 121:
                sols.append((c1,c2,c3))
print(f'\ninteger combos with Fib-part = 121 on rungs (11,9,2), |c|<=3: {sols}')

# Jarlskog cross-link: J = alpha/(2 phi^10); leading order J ~ 1/(sqrt5 phi^20) = 1/(5 F20)
alpha_mp = 1/inv_alpha_ladder
J = alpha_mp / (2 * PHI**10)
print(f'\nJ = alpha(m_p)/(2 phi^10) = {J:.4e}   (PDG: 3.08e-5 +- 0.13e-5 -> {abs(J-3.08e-5)/0.13e-5:.1f} sigma)')
print(f'leading-order pure-golden: 1/(sqrt5 phi^20) = {1/(S5*PHI**20):.4e};  1/(5 F20) = {1/(5*F[20]):.4e}')

# (d/phi)^3 curio
print(f'\n(d/phi)^3 = {(2/PHI)**3:.6f} = 8/phi^3 = {8/PHI**3:.6f}  (the crossing-time coefficient)')
