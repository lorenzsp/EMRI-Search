"""
Profiling script for the EMRI search algorithm implementations.

Measures:
  1. Detection statistic evaluation cost (JAX, baseline)
  2. Trajectory evaluation cost (FEW backend)
  3. PSO swarm fitness evaluation time vs n_particles
  4. TrackOptimizer convergence speed vs current DE + Adam
  5. GPU memory usage under vmap (fewtrax, if available)

Usage
-----
::

    # Basic profiling (FEW backend only):
    python profile_search.py

    # With fewtrax GPU profiling (set FEW_DATA_DIR first):
    FEW_DATA_DIR=/path/to/few/data python profile_search.py --fewtrax

    # Full options:
    python profile_search.py --n-particles 200 --n-iter 5 --fewtrax

Results are printed as a table and saved to ``profiling_results.json``.
"""

import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Profile EMRI search implementations")
parser.add_argument("--n-particles", type=int, default=50,
                    help="Swarm size for PSO profiling (default 50)")
parser.add_argument("--n-iter", type=int, default=5,
                    help="Number of PSO iterations to time (default 5)")
parser.add_argument("--n-repeats", type=int, default=10,
                    help="Number of timing repeats for averaging (default 10)")
parser.add_argument("--fewtrax", action="store_true",
                    help="Enable fewtrax GPU profiling (requires FEW_DATA_DIR env var)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Shared test setup
# ---------------------------------------------------------------------------

def make_test_data(T_data=0.2, T_sft=5e4, deltaT=5.0, snr=30.0, seed=0):
    """Generate a synthetic injection for benchmarking."""
    from emrisearch.search_utils import generate_emri_signal_and_sfts

    rng = np.random.default_rng(seed)
    true_values = np.array([1e6, 10.0, 0.5, 0.2, 0.1, 1.0])
    injection = generate_emri_signal_and_sfts(
        true_values, T_data, T_sft, deltaT, snr, T_data
    )
    return injection, true_values


print("Setting up test data …")
injection, true_values = make_test_data()
data_sfts = jnp.asarray(injection["data_sfts"])
t_obs     = injection["t_alpha"]
T_sft     = 5e4

phi, f, dotf, dotdotf = injection["true_phi_f_fdot_fddot"]
m, n = 2, 0
f_alpha    = jnp.asarray(m * f[0]    + n * f[1])
fdot_alpha = jnp.asarray(m * dotf[0] + n * dotf[1])

results = {}

# ---------------------------------------------------------------------------
# 1. Detection statistic cost
# ---------------------------------------------------------------------------

print("\n[1] Detection statistic evaluation cost")
from emrisearch.jax_utils import det_stat as jax_det_stat

# Warm-up (JIT compile)
_ = jax_det_stat(data_sfts, f_alpha, fdot_alpha, T_sft=T_sft)
jax.block_until_ready(_)

times = []
for _ in range(args.n_repeats):
    t0 = time.perf_counter()
    val = jax_det_stat(data_sfts, f_alpha, fdot_alpha, T_sft=T_sft)
    jax.block_until_ready(val)
    times.append(time.perf_counter() - t0)

stat_ms = np.mean(times) * 1e3
stat_std = np.std(times) * 1e3
print(f"  det_stat()  : {stat_ms:.2f} ± {stat_std:.2f} ms  (n={args.n_repeats})")
results["det_stat_ms"] = stat_ms

# ---------------------------------------------------------------------------
# 2. FEW trajectory evaluation cost
# ---------------------------------------------------------------------------

print("\n[2] FEW trajectory evaluation cost (get_f_fdot_fddot_back)")
from emrisearch.emri_utils import get_f_fdot_fddot_back

params_6d = true_values[np.newaxis, :]

times = []
for _ in range(args.n_repeats):
    t0 = time.perf_counter()
    get_f_fdot_fddot_back(params_6d, t_obs)
    times.append(time.perf_counter() - t0)

traj_ms = np.mean(times) * 1e3
traj_std = np.std(times) * 1e3
print(f"  FEW traj()  : {traj_ms:.2f} ± {traj_std:.2f} ms  (n={args.n_repeats})")
results["few_traj_ms"] = traj_ms

# ---------------------------------------------------------------------------
# 3. PSO swarm fitness evaluation time vs n_particles
# ---------------------------------------------------------------------------

print("\n[3] PSO fitness evaluation: serial FEW (n_particles sweep)")
from emrisearch.pso_utils import ParticleSwarmOptimizer, initialize_swarm_from_track

pso = ParticleSwarmOptimizer(data_sfts, t_obs, T_sft=T_sft)

particle_counts = [10, 25, 50, 100, args.n_particles]
swarm_times = {}
for n_p in particle_counts:
    pos, _, _ = initialize_swarm_from_track(
        0.0, 0.0, 0.2, pso.bounds, n_p, seed=0
    )
    t0 = time.perf_counter()
    pso._evaluate_fitness(pos, mode=(m, n))
    elapsed = time.perf_counter() - t0
    t_per = elapsed / n_p * 1e3
    print(f"  n={n_p:4d}: {elapsed*1e3:.1f} ms total,  {t_per:.2f} ms/particle")
    swarm_times[n_p] = elapsed * 1e3

results["swarm_ms_by_n_particles"] = swarm_times

# ---------------------------------------------------------------------------
# 4. TrackOptimizer (single mode, limited starts) vs raw FEW fitness
# ---------------------------------------------------------------------------

print("\n[4] TrackOptimizer: estimate plunge time + single-mode optimization")
from emrisearch.track_optimizer import TrackOptimizer, estimate_plunge_time

valid = (fdot_alpha > 0) & (f_alpha > 0)
f_obs_np    = np.array(f_alpha)
fdot_obs_np = np.array(fdot_alpha)

t0 = time.perf_counter()
T_pl_est, _ = estimate_plunge_time(f_obs_np, fdot_obs_np)
t_est = (time.perf_counter() - t0) * 1e3
print(f"  estimate_plunge_time()   : {t_est:.2f} ms")
print(f"  T_plunge estimate        : {T_pl_est:.4f} yr  (truth: {true_values[3]:.4f} yr)")
results["T_plunge_estimate_yr"] = T_pl_est
results["T_plunge_truth_yr"]    = float(true_values[3])

# Quick benchmark: 4 starts, 1 mode
optimizer = TrackOptimizer(f_obs_np, fdot_obs_np, t_obs, T_sft=T_sft)
t0 = time.perf_counter()
best_params, best_loss, _ = optimizer.optimize(mode=(m, n), n_starts=4, max_iter=50)
opt_time = (time.perf_counter() - t0) * 1e3
print(f"  TrackOptimizer.optimize() [4 starts]: {opt_time:.1f} ms")
print(f"  Best loss = {best_loss:.3e}")
results["track_optimizer_4starts_ms"] = opt_time

# ---------------------------------------------------------------------------
# 5. fewtrax profiling (optional)
# ---------------------------------------------------------------------------

if args.fewtrax:
    print("\n[5] fewtrax trajectory and batched vmap profiling")
    import os
    data_dir = os.environ.get("FEW_DATA_DIR")
    if data_dir is None:
        print("  ⚠  FEW_DATA_DIR not set; skipping fewtrax profiling.")
    else:
        try:
            import sys
            sys.path.insert(
                0,
                "/Users/bertd/Documents/PhD/LISA/Projects/JAX-waveform/fewtrax/src"
            )
            from fewtrax.data.loader import load_flux_data
            from fewtrax.trajectory import EMRIInspiral

            flux_data = load_flux_data(data_dir)
            traj      = EMRIInspiral(flux_data)

            M, mu, a = 1e6, 10.0, 0.5
            T_pl     = 0.2
            e_f      = 0.1

            # Single trajectory (warm-up + time)
            _ = traj(p0=10.0, e0=e_f, T=T_pl, a=a, M=M, mu=mu,
                     backward=True, e_f=e_f, dense_steps=200)
            jax.block_until_ready(_)

            times = []
            for _ in range(args.n_repeats):
                t0 = time.perf_counter()
                out = traj(p0=10.0, e0=e_f, T=T_pl, a=a, M=M, mu=mu,
                           backward=True, e_f=e_f, dense_steps=200)
                jax.block_until_ready(out)
                times.append(time.perf_counter() - t0)

            single_ms = np.mean(times) * 1e3
            print(f"  Single trajectory (fewtrax): {single_ms:.2f} ms")
            results["fewtrax_single_traj_ms"] = single_ms

            # Batched vmap profiling
            from functools import partial

            @jax.jit
            def batched_traj(params_batch):
                def _one(p):
                    M_, mu_, a_, T_, ef_ = p[0], p[1], p[2], p[3], p[4]
                    return traj(
                        p0=jnp.float64(10.0), e0=ef_, T=T_,
                        a=a_, M=M_, mu=mu_, backward=True, e_f=ef_,
                        dense_steps=100,
                    )
                return jax.vmap(_one)(params_batch)

            for n_batch in [10, 50, 100]:
                params_batch = jnp.tile(
                    jnp.array([M, mu, a, T_pl, e_f], dtype=jnp.float64),
                    (n_batch, 1)
                )
                # Warm-up
                _ = batched_traj(params_batch)
                jax.block_until_ready(_)

                t0 = time.perf_counter()
                out = batched_traj(params_batch)
                jax.block_until_ready(out)
                elapsed_ms = (time.perf_counter() - t0) * 1e3

                print(f"  vmap batch n={n_batch:4d}: "
                      f"{elapsed_ms:.1f} ms  ({elapsed_ms/n_batch:.2f} ms/traj)")
                results[f"fewtrax_vmap_{n_batch}_ms"] = elapsed_ms

        except Exception as exc:
            print(f"  fewtrax profiling failed: {exc}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("PROFILING SUMMARY")
print("=" * 60)
for key, val in results.items():
    if isinstance(val, dict):
        print(f"  {key}:")
        for k2, v2 in val.items():
            print(f"    n={k2}: {v2:.1f} ms")
    else:
        print(f"  {key}: {val}")

out_path = "profiling_results.json"
with open(out_path, "w") as fp:
    json.dump(results, fp, indent=2)
print(f"\nResults saved to {out_path}")
