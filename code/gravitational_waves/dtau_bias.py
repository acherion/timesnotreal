# What fractional damping-time bias does the model's hidden component produce?
# Waveform at degenerate spin: same frequency, amplitude A = 2/phi^7 = 6.9%,
# slower decay tau_c = k*tau. Envelope (phasor sum):
#   E(t) = | e^{-t/tau} + A e^{i psi} e^{-t/(k tau)} |
# Fit a single exponential to E(t), SNR-weighted (w = E^2), over window [0, T].
# Report delta_tau = tau_fit/tau - 1 across psi and k. Also the component's own
# SNR relative to the ringdown SNR (bounded by long-lived-line searches).
import math

PHI = (1 + 5**0.5) / 2
A = 2 / PHI**7

def fit_dtau(k, psi, T, n=4000):
    # weighted least squares of ln E(t) vs (b - t/tauf), weights E^2
    S = Sx = Sy = Sxx = Sxy = 0.0
    for i in range(n):
        t = T * (i + 0.5) / n
        re = math.exp(-t) + A * math.cos(psi) * math.exp(-t / k)
        im = A * math.sin(psi) * math.exp(-t / k)
        E = math.hypot(re, im)
        w = E * E
        y = math.log(E)
        S += w; Sx += w * t; Sy += w * y; Sxx += w * t * t; Sxy += w * t * y
    slope = (S * Sxy - Sx * Sy) / (S * Sxx - Sx * Sx)   # = -1/tau_fit
    return -1 / slope - 1

def comp_snr_ratio(k):
    # SNR of child component / SNR of main ringdown (same frequency, matched filter)
    #   SNR ~ A * sqrt( integral e^{-2t/k} / integral e^{-2t} ) = A*sqrt(k)
    return A * math.sqrt(k)

print(f'A = 2/phi^7 = {A:.4f}')
print()
print('delta_tau (fractional damping-time bias) from SNR-weighted single-exp fit:')
print('  k=tau_c/tau   T=4tau: psi=0    pi/2     pi     3pi/2   | psi-avg | T=6tau avg')
for k in (2, 3, 5, 8):
    row = [fit_dtau(k, p * math.pi / 2, 4) for p in range(4)]
    avg4 = sum(row) / 4
    avg6 = sum(fit_dtau(k, p * math.pi / 2, 6) for p in range(4)) / 4
    print(f'   {k:4.1f}        {row[0]:+7.3f}  {row[1]:+7.3f}  {row[2]:+7.3f}  {row[3]:+7.3f}  | {avg4:+7.3f} | {avg6:+7.3f}')
print()
print('child-component SNR as fraction of ringdown SNR (long-line searches bound this):')
for k in (2, 3, 5, 8, 20, 100):
    r = comp_snr_ratio(k)
    print(f'   k={k:5.0f}:  SNR_child/SNR_ringdown = {r:.3f}'
          f'   -> for GW250114 ringdown SNR ~ 30: SNR_child ~ {30*r:.1f} (limit ~ 6.8)')
