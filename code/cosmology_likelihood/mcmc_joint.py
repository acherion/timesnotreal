import os, sys, time, json
# Posterior sampling (emcee) for LCDM and the locked model on the joint likelihood
# (Planck native plik-lite TTTEEE + lowl TT + lowl EE + DESI DR2 BAO + SN shape), flat priors over the fit bounds,
# Gaussian prior on A_planck. Long-running background job; chains saved to HDF5 (emcee backend) every step.
# Usage: python mcmc_joint.py lcdm|model [nsteps]
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, emcee
sys.argv_case = sys.argv[1]
import importlib.util
spec = importlib.util.spec_from_file_location('lap', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'laplace_lib.py'))
lap = importlib.util.module_from_spec(spec); spec.loader.exec_module(lap)
CASE = sys.argv[1]; NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
lap.CASE = CASE
if CASE == 'lcdm':
    x0 = np.array([0.022516, 0.117772, 68.330, 3.04265, 0.97047, 0.055975, 1.000785])
    lo = np.array([0.019, 0.08, 60, 2.9, 0.93, 0.02, 0.98]); hi = np.array([0.026, 0.16, 76, 3.2, 1.0, 0.12, 1.02])
    sig = np.array([1.5e-4, 1.2e-3, 0.5, 0.015, 0.004, 0.007, 0.0025])
else:
    x0 = np.array([0.022769, 66.509, 3.03805, 0.056291, 1.000793])
    lo = np.array([0.019, 60, 2.9, 0.02, 0.98]); hi = np.array([0.026, 76, 3.2, 0.12, 1.02])
    sig = np.array([1.1e-4, 0.07, 0.015, 0.007, 0.0025])
ndim = len(x0); nwalkers = 4*ndim
def logp(x):
    if np.any(x < lo) or np.any(x > hi): return -np.inf
    try: c2 = lap.chi2(x)
    except Exception: return -np.inf
    if not np.isfinite(c2): return -np.inf
    return -0.5*c2
backend = emcee.backends.HDFBackend(f'mcmc_{CASE}.h5'); 
if not os.path.exists(f'mcmc_{CASE}.h5') or backend.iteration == 0:
    backend.reset(nwalkers, ndim); p0 = x0 + sig*np.random.default_rng(1).normal(size=(nwalkers, ndim))
else:
    p0 = None
sampler = emcee.EnsembleSampler(nwalkers, ndim, logp, backend=backend)
t0 = time.time(); log = open(f'mcmc_{CASE}.log', 'a', encoding='utf-8')
log.write(f'=== start {CASE} walkers={nwalkers} steps={NSTEPS} {time.ctime()}\n'); log.flush()
for i, _ in enumerate(sampler.sample(p0, iterations=NSTEPS, progress=False)):
    if (sampler.iteration % 25) == 0:
        lp = sampler.get_log_prob(flat=True); ch = sampler.get_chain(flat=True)
        try: tau = sampler.get_autocorr_time(tol=0)
        except Exception: tau = None
        log.write(f'iter {sampler.iteration:5d}  best chi2 {-2*lp.max():.3f}  mean acc {np.mean(sampler.acceptance_fraction):.3f}  tau {None if tau is None else np.round(tau,1).tolist()}  {(time.time()-t0)/60:.1f} min\n'); log.flush()
lp = sampler.get_log_prob(flat=True); ch = sampler.get_chain(flat=True)
log.write(f'DONE {time.ctime()} best chi2 {-2*lp.max():.3f}\n'); log.flush()
