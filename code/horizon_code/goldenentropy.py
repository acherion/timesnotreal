# Black-hole entropy corrections from Fibonacci-anyon horizon counting.
# Question: S(A) = A/4 + beta*ln A + const + ...   What is beta for our model?
# Benchmarks: LQG (SU(2) Chern-Simons, Kaul-Majumdar): beta = -3/2
#             U(1) counting: beta = -1/2
# We count EXACTLY (integer arithmetic) and fit.
import math
from fractions import Fraction

PHI = (1 + 5**0.5) / 2
LNPHI = math.log(PHI)

# ---------------------------------------------------------------
# 1. Open chain of n tau-anyons fusing to vacuum (sphere topology)
#    dim = paths on fusion graph 1 -> ... -> 1 with n tau steps
#    Fusion graph adjacency (states {1, tau}): N[a][b] = # ways a x tau -> b
N = [[0, 1], [1, 1]]  # 1 x tau = tau ; tau x tau = 1 + tau

def chain_vacuum_dim(n):
    v = [1, 0]  # start in vacuum
    for _ in range(n):
        v = [v[0]*N[0][0] + v[1]*N[1][0], v[0]*N[0][1] + v[1]*N[1][1]]
    return v[0]  # end in vacuum

# 2. Ring of n cells, golden-mean constraint (no two adjacent tau) = Lucas numbers
def lucas(n):
    a, b = 2, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# integer log via math.log of big ints (Python handles exactly enough via log2)
def ln_big(x):
    return math.log2(x) * math.log(2)

# ---------------------------------------------------------------
# Fit S(n) = alpha*n + beta*ln n + c on windows, exact 3x3 solves
def fit_window(ns, Ss):
    import itertools
    # least squares over the window
    import statistics
    rows = [(n, math.log(n), 1.0) for n in ns]
    # normal equations
    A = [[sum(r[i]*r[j] for r in rows) for j in range(3)] for i in range(3)]
    b = [sum(rows[k][i]*Ss[k] for k in range(len(ns))) for i in range(3)]
    # solve 3x3
    import copy
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(3):
            if r != col:
                f = M[r][col]/M[col][col]
                M[r] = [M[r][j] - f*M[col][j] for j in range(4)]
    return [M[i][3]/M[i][i] for i in range(3)]

print('=== 1. Open chain, n tau punctures -> vacuum (sphere) ===')
ns = list(range(50, 801, 50))
Ss = [ln_big(chain_vacuum_dim(n)) for n in ns]
a, be, c = fit_window(ns, Ss)
print(f'  fit S = a*n + beta*ln n + c over n=50..800:')
print(f'  a       = {a:.12f}   (ln phi = {LNPHI:.12f})')
print(f'  beta    = {be:.3e}   <-- log-correction coefficient')
print(f'  const   = {c:.10f}   compare -ln(sqrt5*phi^2)= {-math.log(5**0.5*PHI**2):.10f}'
      f'  -ln(sqrt5)= {-math.log(5**0.5):.10f}  -ln(sqrt5*phi)= {-math.log(5**0.5*PHI):.10f}')

print()
print('=== 2. Ring of n cells, golden-mean strings (the doc horizon channel) ===')
for n in [50, 100, 200, 400]:
    S = ln_big(lucas(n))
    print(f'  n={n:4d}: ln L_n - n*ln(phi) = {S - n*LNPHI: .3e}')
print('  -> S = n ln(phi) EXACTLY up to exponentially small terms: beta = 0, const = 0')

print()
print('=== 3. Charge-projected ring: n tau anyons on a circle -> vacuum ===')
# closed fusion paths: Tr restricted = (N^n)[1][1] with periodic identification
def ring_vacuum_dim(n):
    # number of closed paths of length n on fusion graph starting/ending at vacuum
    return chain_vacuum_dim(n)  # same as chain 1->1
ns = list(range(50, 801, 50))
Ss = [ln_big(ring_vacuum_dim(n)) for n in ns]
a, be, c = fit_window(ns, Ss)
print(f'  a = {a:.12f}, beta = {be:.3e}, const = {c:.10f}')

print()
print('=== 4. Area-fluctuating ensemble (do we generate a log term?) ===')
# Punctures carry species with different integer area quanta:
#   tau: area 2, tau-bar: area 2, tau*tau-bar: area 3 (toy quanta; ordered punctures)
# Count sequences with total area A and overall vacuum fusion charge (Fibonacci sector).
# DP over (area, fusion state). Fusion state in {1, tau} for the Fib component.
AMAX = 4000
species = [('tau', 2), ('taubar', 2), ('both', 3)]  # each acts as tau-step in Fib sector
dp = [[0, 0] for _ in range(AMAX + 1)]
dp[0][0] = 1
for A in range(AMAX + 1):
    for s in (0, 1):
        if dp[A][s] == 0:
            continue
        for name, da in species:
            if A + da <= AMAX:
                # tau-type step moves fusion state via N
                dp[A + da][0] += dp[A][s] * N[s][0] * 0 + (dp[A][s] if N[s][0] else 0)*0
    # (rewrite below properly)
# redo cleanly
dp = [[0, 0] for _ in range(AMAX + 1)]
dp[0][0] = 1
for A in range(AMAX):
    v0, v1 = dp[A]
    if v0 == 0 and v1 == 0:
        continue
    for name, da in species:
        if A + da > AMAX:
            continue
        # applying a tau line: (v0, v1) -> (v1, v0 + v1)
        dp[A + da][0] += v1
        dp[A + da][1] += v0 + v1
counts = [(A, dp[A][0]) for A in range(100, AMAX + 1) if dp[A][0] > 0]
As = [A for A, cnt in counts if A >= 1000 and A % 100 == 0]
Ss = [ln_big(dict(counts)[A]) for A in As]
a, be, c = fit_window(As, Ss)
print(f'  ordered punctures, mixed areas, vacuum charge, A=1000..4000:')
print(f'  a = {a:.10f}, beta = {be:.3e}, const = {c:.6f}')
print('  -> simple-pole generating function: still NO log term (beta ~ 0)')

print()
print('=== 5. Canonical -> microcanonical inversion (the only -1/2 source) ===')
# If the horizon exchanges cells with environment (n fluctuates, Gaussian width ~ sqrt(A)),
# S_micro(A) = S_can - (1/2) ln(2 pi sigma^2), sigma^2 = kappa*A  =>  beta = -1/2.
print('  n fluctuates Gaussianly with sigma^2 prop A  =>  S = A/4 - (1/2) ln A + const')

print()
print('=== 6. Fixed constants of the theory ===')
D_doubled = 1 + PHI**2
print(f'  total quantum dimension doubled Fib: D = 1+phi^2 = {D_doubled:.10f} = phi*sqrt5 = {PHI*5**0.5:.10f}')
print(f'  topological entanglement entropy gamma = ln D = {math.log(D_doubled):.10f}'
      f' = ln phi + ln sqrt5 = {LNPHI + math.log(5**0.5):.10f}')
print(f'  cell area from n ln phi = A/4: a_cell = 4 ln phi = {4*LNPHI:.10f} lP^2')
print(f'  Immirzi-style check: QGR claim 2*pi*gamma_I = phi -> gamma_I = {PHI/(2*math.pi):.6f}')
print(f'  (LQG values: Meissner 0.23753, Ghosh-Mitra ~0.274)')
