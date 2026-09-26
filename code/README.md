# Reproducibility package — timesnotreal.com

This directory contains the analysis scripts behind the quantitative claims on the site,
including the ones that came out against the model. Everything here is runnable with public
data and open-source codes. Results are maximum-likelihood optimizations (Nelder–Mead), not
posterior samples, unless stated.

## The model in ten lines (what the scripts assume)

- Two structural inputs: the golden ratio φ and d = 2 (EM polarizations). No free cosmological parameters.
- Extraction fraction p = 2/φ⁷ = 0.06888; recursion depth today n₀ = 2φ² = 5.236.
- Present-day fractions: Ω_DE = (1−p)^n₀ = 0.688, Ω_DM = Ω_DE/φ² = 0.263, Ω_b = 1 − Ω_DE(3−φ) = 0.0490. **Ω_m = 0.312 is locked.**
- Dark energy: d ln ρ_DE/dt = −3·C·H₀ with C = n₀|ln(1−p)|/(3H₀t₀) = 0.133, i.e. w(z) = −1 + 0.133/E(z),
  E = H/H₀, solved self-consistently. Never crosses −1. Matter dilutes as a⁻³ (no energy transfer).
- Primordial tilt n_s = 1 − 1/φ⁷ = 0.9656 (fixed). Optional infrared cutoff k_min = 2.3×10⁻⁴ Mpc⁻¹.
- Two dimensionful anchors: H₀ (or t₀) and α(m_p). 1/α(m_p) = (φ¹¹ + φ⁹ − 2φ²)/2 = 134.89 is a structured
  closed form whose coefficients are **not** derived. G = (k_e e²/m_p²) α^{4φ³} is a consistency relation, 0.2% high.
- The black-hole-interior picture (Part III.5 of the site) is an interpretive overlay; the cosmology scripts
  do not depend on it except through the k_min cutoff and the ringdown scripts.

## Environment

```
pip install numpy scipy camb cobaya
pip install --use-pep517 "git+https://github.com/ACTCollaboration/DR6-ACT-lite.git"
python -m cobaya install planck_2018_highl_plik.TTTEEE_lite_native planck_2018_lowl.TT planck_2018_lowl.EE act_dr6_cmbonly -p ./cobaya_packages --no-set-global
set COBAYA_PACKAGES_PATH=./cobaya_packages     (or export on Unix)
```
Run scripts from their own directory (log files are written to the working directory).
Versions used: camb 2.0.3, cobaya 3.6.2, ACT-DR6-CMBonly 1.0.0, Python 3.12, Windows.

## What each script does

### cosmology_likelihood/
| script | computes |
|---|---|
| `plikfit.py lcdm|model|modelfree` | Maximizes Planck 2018 native likelihoods (plik-lite TTTEEE + low-ℓ TT + low-ℓ EE) for ΛCDM, the locked model, and the model's w(z) with Ω_m and n_s unlocked. |
| `jointfull.py lcdm|model` | Same Planck likelihoods + DESI DR2 BAO (13 points, per-tracer DM–DH correlations) + SN shape block (Ω_m = 0.330 ± 0.015 compression). |
| `actfit.py lcdm|model` | ACT DR6 CMB-only (ℓ = 600–6500) + Planck low-ℓ EE, with ~1%/2% calibration priors. |
| `actjoint.py lcdm|model` | ACT DR6 + low-ℓ EE + BAO + SN. The A_L-anomaly-independent joint verdict. |
| `followups.py` | At the Planck best fits: k_min cutoff test on low-ℓ TT; A_lens scan. |
| `jointfit.py` | Fast version with Planck compressed distance priors (R, ℓ_A, ω_b) + BAO; four cases incl. free N_eff and Y_He. |
| `threeprobes.py` | Adds SN blocks for four compilations (Pantheon+, DES-SN5YR, DES-Dovekie, Union3). |
| `coupled.py`, `fscan.py` | Partial DE→DM energy-transfer variant; scan of the transfer fraction f. |
| `junction.py` | Junction-window variants (drain frozen or delayed above z_j). |
| `boltzmann.py`, `verify_de.py`, `run5.py` | First CAMB runs: R, θ*, σ₈, low-ℓ cutoff signature; DE-table validation. |

### mock_and_probes/
| script | computes |
|---|---|
| `mockfit.py`, `cplfit.py`, `robust.py` | Mock-universe discordance test: ΛCDM fitters applied to synthetic model-universe BAO/SN/CMB data. |
| `transfer.py` | Literal fuel→matter transfer vs BAO shape (excluded). |
| `growth.py`, `growth_corner.py`, `fsigma8.py` | fσ₈(z) vs 13-point RSD compilation; S₈ at the model's corners; lensing-potential ratio (A_L proxy). |
| `isw.py`, `isw_cross.py` | Late-ISW low-ℓ TT enhancement; ISW–galaxy cross-correlation amplitude ratio vs redshift. |
| `kmin_cutoff.py`, `glitch_scan.py`, `jwst_age.py` | Cutoff scale from the parent mass; DESI BAO step-feature scan (null); age at high z (negative result). |

### background_and_identities/
Dimensionless identities, the boundary-crossing (timestamp) analysis, far-future integration, Kerr-horizon
oblateness spin-meter, α-formula algebra (D² reduction, Binet form, uniqueness scan).

### gravitational_waves/
Damping-time bias of a hidden 6.9% component (`dtau_bias.py`), remnant-spin census (`spin_census.py`),
Davies point (`davies_point.py`), Foit–Kleban line spacing (`foitkleban.py`).

### horizon_code/
Golden-chain (Fibonacci anyon) diagonalization, central charges, ground-state energy density, momentum-resolved
conformal towers (`chain_lorentz.py`), and the exact horizon entropy count (`goldenentropy.py`).

## Results table (September 2026) — including failures

| test | ΛCDM | locked model | verdict |
|---|---|---|---|
| Planck 2018 native (plik-lite TTTEEE + lowTT + lowEE), −2lnL | 1003.0 (H₀ 67.3, θ* 1.04109) | 1041.0 (H₀ 66.5, θ* 1.04167, σ₈ 0.759) | **model +38.0**, two fewer parameters |
| same, model w(z) with Ω_m, n_s unlocked | — | 1004.0 (H₀ 64.3, Ω_m 0.347) | +1.0: the Ω_m lock is the entire cost |
| Planck + DESI DR2 BAO + SN, joint | 1022.8 (CMB 1006.3, BAO 12.8, SN 3.7) | 1054.9 (CMB 1040.9, BAO 11.1, SN 3.0) | **model +32.0** |
| ACT DR6 + lowEE, CMB-only | 546.4 (degenerate corner H₀ 66.0, Ω_m 0.34) | 571.1 | model +24.8 |
| ACT DR6 + lowEE + BAO + SN, joint | 571.8 (H₀ 68.4, Ω_m 0.301) | 584.9 (H₀ 66.5) | **model +13.2** (A_L-anomaly-independent) |
| A_lens scan on Planck | prefers ~1.1 (−5) | prefers ~1.2 (−17) | ~12–15 of the Planck deficit is the lensing anomaly |
| compressed distance priors + BAO (fast) | 14.7 | 28.3 | +13.6 (understates the full likelihood) |
| + free N_eff ≥ 3.044 / free Y_He | — | slides back to boundaries | early-universe escape closed |
| junction variants (drain frozen/delayed above z_j) | — | best −2.4 | high-z tail escape closed |
| partial DE→DM transfer, fraction f | — | monotonically worse (f=0.05: +7, f=0.10: +16) | coupled-sector escape closed |
| k_min cutoff on Planck low-ℓ TT | −1.4 | −1.5 | mild preference for the cutoff, both cosmologies |
| fσ₈ vs 13-pt RSD, χ² | 10.6 | 9.5–9.9 (own anchors), 12–13 (CMB corner) | draw / mild preference at own anchors |
| S₈ | 0.832 | 0.815 (own), 0.810 (Planck params), 0.774–0.785 (CMB corner) | anchor-dependent |
| lensing-potential power ratio | 1 | 0.970 (Planck params), ~0.85 (CMB corner) | A_L ≈ 0.97, headwind vs Planck's A_L > 1 |
| ISW–galaxy cross amplitude ratio | 1 | 1.04 (z=0.3) → 1.20 (z=1.0) → 1.27 (z=1.5) | committed prediction, rising with z |
| mock-universe SN-shape Ω_m read by ΛCDM | — | 0.35–0.36 | real compilations 0.330–0.356: ordering retrodicted, amplitude ~1.5σ high |
| mock BAO+BBN vs CMB apparent H₀ | — | 68.2 vs 67.3 | real 68.5 vs 67.4: retrodicted |
| DESI BAO-alone apparent Ω_m | — | 0.314 | real 0.2975 ± 0.0086: 1.9σ unexplained |
| G relation | 6.6743 ± 0.00015 ×10⁻¹¹ | 6.687 ×10⁻¹¹ | 0.2% high = 87σ as exact claim; = 0.012% in α(m_p) |
| Kerr remnant spin coincidences | — | Q = 2φ at a = 0.6856 (0.11% from 0.6865); Mω = φ³/8 at 0.693 | near, not exact; no exact coincidence claimed |
| GWTC-5.0 damping-time residual | GR = 0 | forecast +2 to +8% | measured joint +7 (+6/−5)%: inside band |

Retractions kept visible on the site: sustained-accretion lockstep (Aug 2026); r_d ≈ 145 Mpc reading (retracted within a day);
G₂ as Lagrangian symmetry (Sept 2026); octonion non-associativity as the source of irreversibility (Sept 2026).

## Known limitations of these analyses

- Optimizations, not MCMC; likelihood values at best fit only. Nelder–Mead convergence tolerance ~0.02–0.05.
- Plik-lite and ACT-lite are foreground-marginalized ("lite") likelihoods.
- BAO and SN blocks are Gaussian compressions; DESI DR2 central values are as tabulated in the scripts.
- RSD compilation uses diagonal errors (BOSS/WiggleZ covariances ignored).
- The model's w(a) table is built by an external iteration and passed to CAMB's PPF dark-energy module.
