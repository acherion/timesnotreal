# Remnant-spin census from GWOSC catalog data.
# Final spin via Barausse-Rezzolla (2009) aligned-spin formula, a1=a2=chi_eff.
# Census question: how tightly do comparable-mass remnants cluster at 0.6865
# (the model's degenerate spin), and which loud events sit OFF the degeneracy
# (targets where the child line would be frequency-resolvable)?
import json, math

PHI = (1 + 5**0.5) / 2
s4, s5, t0, t2, t3 = -0.1229, 0.4537, -2.8904, -3.5171, 2.5763

def final_spin(m1, m2, chi):
    if m2 > m1: m1, m2 = m2, m1
    q = m2 / m1
    nu = q / (1 + q)**2
    a1 = a2 = chi
    asum = a1*a1 + a2*a2*q**4 + 2*a1*a2*q*q
    ell = (s4/(1+q*q))*asum + ((s5*nu + t0 + 2)/(1+q*q))*(a1 + a2*q*q) \
          + 2*math.sqrt(3) + t2*nu + t3*nu*nu
    return math.sqrt(asum + 2*(a1 + a2*q*q)*ell*q + ell*ell*q*q) / (1+q)**2

def Momega220(a):     # Berti et al. fit for Kerr (2,2,0)
    return 1.5251 - 1.1568 * (1 - a)**0.1292

print(f'check: equal-mass nonspinning -> a_f = {final_spin(1,1,0):.4f} (expect 0.6865)')

d = json.load(open(r'gwtc.json'))
rows = []
for name, e in d['events'].items():
    m1, m2, chi, snr = e['mass_1_source'], e['mass_2_source'], e['chi_eff'], e['network_matched_filter_snr']
    if None in (m1, m2, chi, snr): continue
    if m2 < 3: continue                      # skip NS secondaries handled separately
    af = final_spin(m1, m2, chi)
    rows.append((e['commonName'], snr, m2/m1 if m1>m2 else m1/m2, chi, af))

# de-duplicate by commonName keeping highest SNR entry
best = {}
for r in rows:
    if r[0] not in best or r[1] > best[r[0]][1]: best[r[0]] = r
rows = sorted(best.values(), key=lambda r: -r[1])
print(f'\n{len(rows)} BBH events with (m1, m2, chi_eff, SNR)')

for label, sel in [('ALL BBH', rows),
                   ('comparable-mass (q>0.7), modest spin (|chi_eff|<0.2)',
                    [r for r in rows if r[2] > 0.7 and abs(r[3]) < 0.2]),
                   ('loud subset (SNR>12) of the above',
                    [r for r in rows if r[2] > 0.7 and abs(r[3]) < 0.2 and r[1] > 12])]:
    afs = sorted(r[4] for r in sel)
    n = len(afs)
    med = afs[n//2]
    mean = sum(afs)/n
    sd = (sum((a-mean)**2 for a in afs)/n)**0.5
    within = sum(1 for a in afs if abs(a - 0.6865) < 0.02)
    print(f'\n  {label}: n={n}')
    print(f'    median a_f = {med:.4f}   mean = {mean:.4f}   sd = {sd:.4f}')
    print(f'    within +-0.02 of 0.6865: {within}/{n} = {100*within/n:.0f}%')

print('\nLoud off-degeneracy targets (SNR>=13, |a_f - 0.6865| > 0.08):')
print('  event               SNR    q     chi_eff   a_f    f_child/f_220')
for r in rows:
    if r[1] >= 13 and abs(r[4] - 0.6865) > 0.08:
        ratio = PHI**3 / (8 * Momega220(r[4]))
        print(f'  {r[0]:<18} {r[1]:5.1f}  {r[2]:.2f}   {r[3]:+.2f}    {r[4]:.3f}   {ratio:.3f}')
