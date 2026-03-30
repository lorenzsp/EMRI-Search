"""
JAX-accelerated Particle Swarm Optimization for EMRI parameter recovery.

The swarm evaluates the semi-coherent detection statistic as its fitness
function and is initialized with a focused distribution around the plunge-time
estimate derived from a single STFT-identified track.

PSO hyperparameters default to the Clerc-Kennedy constriction factor, which
guarantees convergence::

    w  = 0.729   (inertia)
    c1 = 1.494   (cognitive coefficient)
    c2 = 1.494   (social coefficient)

Usage
-----
::

    from emrisearch.pso_utils import ParticleSwarmOptimizer

    pso = ParticleSwarmOptimizer(data_sfts, t_obs, T_sft=5e4)
    best_params, best_fitness, history = pso.optimize(
        mode=(2, 0),
        n_particles=200,
        n_iter=100,
        T_plunge_estimate=0.9,   # years, from STFT estimate
    )
    # best_params = [M, mu, a, T_plunge, e_f]
"""

from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .draw_population import ParameterScaler
from .track_optimizer import estimate_plunge_time, DEFAULT_BOUNDS


# ---------------------------------------------------------------------------
# Swarm initialization
# ---------------------------------------------------------------------------

def initialize_swarm_from_track(f0: float,
                                  fdot0: float,
                                  T_plunge_estimate: float,
                                  bounds: np.ndarray,
                                  n_particles: int = 200,
                                  seed: int = 42) -> tuple:
    """Initialize a PSO swarm with constraints from an STFT anchor track.

    Narrows the T_plunge bounds around the estimate and uses Sobol
    quasi-random sampling for low-discrepancy coverage.

    Parameters
    ----------
    f0 : float
        Representative observed GW frequency [Hz].
    fdot0 : float
        Representative observed frequency derivative [Hz/s].
    T_plunge_estimate : float
        Estimated time to plunge [years].
    bounds : np.ndarray, shape (5, 2)
        Physical parameter bounds ``[M, mu, a, T_plunge, e_f]``.
    n_particles : int
        Swarm size.
    seed : int
        Random seed for Sobol sampler.

    Returns
    -------
    positions : np.ndarray, shape (n_particles, 5)
        Initial particle positions.
    velocities : np.ndarray, shape (n_particles, 5)
        Initial particle velocities (small random perturbations).
    effective_bounds : np.ndarray, shape (5, 2)
        Narrowed bounds after applying the T_plunge constraint.
    """
    T_plunge_width = 0.25  # ± years around the estimate
    effective_bounds = bounds.copy()
    effective_bounds[3, 0] = max(bounds[3, 0], T_plunge_estimate - T_plunge_width)
    effective_bounds[3, 1] = min(bounds[3, 1], T_plunge_estimate + T_plunge_width)

    scaler = ParameterScaler(effective_bounds, seed=seed)
    positions = scaler.draw_samples(n_particles)

    rng = np.random.default_rng(seed)
    ranges = effective_bounds[:, 1] - effective_bounds[:, 0]
    velocities = rng.uniform(-0.05, 0.05, (n_particles, 5)) * ranges[np.newaxis, :]

    return positions, velocities, effective_bounds


# ---------------------------------------------------------------------------
# JAX PSO update step
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=('n_dims',))
def pso_update_step(positions: jnp.ndarray,
                    velocities: jnp.ndarray,
                    personal_best_pos: jnp.ndarray,
                    personal_best_fit: jnp.ndarray,
                    global_best_pos: jnp.ndarray,
                    key: jax.Array,
                    w: float = 0.729,
                    c1: float = 1.494,
                    c2: float = 1.494,
                    lower_bounds: jnp.ndarray | None = None,
                    upper_bounds: jnp.ndarray | None = None,
                    n_dims: int = 5) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One JAX-jitted PSO velocity and position update.

    Implements the standard Clerc-Kennedy constricted PSO::

        v ← w·v + c₁·r₁·(pbest − x) + c₂·r₂·(gbest − x)
        x ← x + v

    Particles that hit a boundary have their velocity zeroed in the
    corresponding dimension (absorbing boundary condition).

    Parameters
    ----------
    positions : jnp.ndarray, shape (n_particles, n_dims)
        Current particle positions.
    velocities : jnp.ndarray, shape (n_particles, n_dims)
        Current particle velocities.
    personal_best_pos : jnp.ndarray, shape (n_particles, n_dims)
        Personal best positions.
    personal_best_fit : jnp.ndarray, shape (n_particles,)
        Personal best fitness values (unused here; included for API symmetry).
    global_best_pos : jnp.ndarray, shape (n_dims,)
        Global best position.
    key : jax.Array
        JAX random key.
    w, c1, c2 : float
        PSO hyperparameters (Clerc-Kennedy defaults).
    lower_bounds, upper_bounds : jnp.ndarray, optional
        Boundary arrays of shape (n_dims,).
    n_dims : int
        Number of dimensions (static, required for JIT shape inference).

    Returns
    -------
    new_positions : jnp.ndarray, shape (n_particles, n_dims)
    new_velocities : jnp.ndarray, shape (n_particles, n_dims)
    """
    key1, key2 = jax.random.split(key)
    n_particles = positions.shape[0]

    r1 = jax.random.uniform(key1, (n_particles, n_dims))
    r2 = jax.random.uniform(key2, (n_particles, n_dims))

    cognitive = c1 * r1 * (personal_best_pos - positions)
    social    = c2 * r2 * (global_best_pos[None, :] - positions)

    new_velocities = w * velocities + cognitive + social
    new_positions  = positions + new_velocities

    if lower_bounds is not None and upper_bounds is not None:
        # Absorbing boundary: zero velocity when crossing a wall
        at_lower = new_positions < lower_bounds[None, :]
        at_upper = new_positions > upper_bounds[None, :]
        new_velocities = jnp.where(at_lower | at_upper, 0.0, new_velocities)
        new_positions  = jnp.clip(new_positions, lower_bounds, upper_bounds)

    return new_positions, new_velocities


# ---------------------------------------------------------------------------
# ParticleSwarmOptimizer
# ---------------------------------------------------------------------------

class ParticleSwarmOptimizer:
    """PSO optimizer for EMRI parameter recovery via the semi-coherent statistic.

    The fitness function is the JAX detection statistic
    ``jax_utils.det_stat(data_sfts, f_alpha, fdot_alpha, T_sft=...)``,
    which requires a frequency track ``(f_alpha, fdot_alpha)`` evaluated
    from the EMRI trajectory.

    Each PSO iteration:

    1. Calls ``emri_utils.get_f_fdot_fddot_back()`` for each particle to
       obtain ``(f, fdot)`` at the observation times.
    2. Evaluates ``jax_utils.det_stat()`` on the resulting track.
    3. Updates velocities and positions with ``pso_update_step()``.

    When ``fewtrax`` is available, step 1 can be replaced by a single
    ``jax.vmap`` call over ``fewtrax.EMRIInspiral`` (see the
    ``use_fewtrax`` parameter), which is O(10–100×) faster.

    Parameters
    ----------
    data_sfts : array-like, shape (n_freq, n_sft)
        SFT data array.
    t_obs : array-like, shape (n_sft,)
        SFT mid-times [s].
    T_sft : float
        SFT segment duration [s].
    bounds : array-like, optional
        Parameter bounds, shape (5, 2).
    use_fewtrax : bool
        If True, attempt to use fewtrax for batched trajectory evaluation
        (requires fewtrax to be installed and ``flux_data`` to be provided).
    flux_data : optional
        fewtrax FluxData object.  Required when ``use_fewtrax=True``.
    """

    def __init__(self,
                 data_sfts,
                 t_obs: np.ndarray,
                 T_sft: float = 5e4,
                 bounds: np.ndarray | None = None,
                 use_fewtrax: bool = False,
                 flux_data=None):
        from .jax_utils import det_stat as _jax_det_stat

        self.data_sfts = jnp.asarray(data_sfts)
        self.t_obs     = np.asarray(t_obs, dtype=float)
        self.T_sft     = float(T_sft)
        self._det_stat = _jax_det_stat

        if bounds is None:
            self.bounds = DEFAULT_BOUNDS.copy()
        else:
            self.bounds = np.asarray(bounds, dtype=float).copy()

        self.use_fewtrax = use_fewtrax and (flux_data is not None)
        self.flux_data   = flux_data

        if use_fewtrax and flux_data is None:
            import warnings
            warnings.warn(
                "use_fewtrax=True but flux_data is None; falling back to FEW backend."
            )

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _fitness_few(self, params_5d: np.ndarray, mode: tuple) -> float:
        """Evaluate negative detection statistic using the FEW backend."""
        from .emri_utils import get_f_fdot_fddot_back

        m, n = mode
        params_6d = np.append(params_5d, 1.0)[np.newaxis, :]
        try:
            _, f, fdot, _ = get_f_fdot_fddot_back(params_6d, self.t_obs)
            f_alpha    = jnp.array(m * f[0]    + n * f[1],    dtype=jnp.float64)
            fdot_alpha = jnp.array(m * fdot[0] + n * fdot[1], dtype=jnp.float64)
            stat = self._det_stat(
                self.data_sfts, f_alpha, fdot_alpha, T_sft=self.T_sft
            )
            return -float(stat)
        except Exception:
            return 1e10

    def _batch_fitness_fewtrax(self,
                                 positions: np.ndarray,
                                 mode: tuple) -> np.ndarray:
        """Evaluate negative detection statistic for all particles via fewtrax.

        Uses ``jax.vmap`` over ``fewtrax.EMRIInspiral`` for batched trajectory
        evaluation on GPU, then ``jax.vmap`` over the detection statistic.

        Parameters
        ----------
        positions : np.ndarray, shape (n_particles, 5)
        mode : tuple (m, k, n) or (m, n)

        Returns
        -------
        fitness : np.ndarray, shape (n_particles,)
            Negative detection statistic for each particle.
        """
        from fewtrax.trajectory import EMRIInspiral
        from fewtrax.utils.geodesic import get_fundamental_frequencies
        from fewtrax.utils.constants import YEAR_SI, MTSUN_SI

        m, k, n = mode if len(mode) == 3 else (mode[0], 0, mode[1])
        traj = EMRIInspiral(self.flux_data)
        t_obs_jax = jnp.array(self.t_obs, dtype=jnp.float64)
        data_sfts = self.data_sfts
        T_sft     = self.T_sft
        det_stat  = self._det_stat

        def _single_particle_fitness(params):
            M, mu, a, T_plunge, e_f = (
                params[0], params[1], params[2], params[3], params[4]
            )
            t_back, p_back, e_back, _, _, _ = traj(
                p0=jnp.float64(10.0), e0=e_f, T=T_plunge,
                a=a, M=M, mu=mu, backward=True, e_f=e_f, dense_steps=200,
            )
            M_total_s = (M + mu) * MTSUN_SI

            def freq_one(p, e):
                Om_phi, Om_theta, Om_r = get_fundamental_frequencies(
                    jnp.abs(a), p, e, 1.0
                )
                return jnp.abs(m * Om_phi + k * Om_theta + n * Om_r) / (
                    2.0 * jnp.pi * M_total_s
                )

            f_track    = jax.vmap(freq_one)(p_back, e_back)
            fdot_track = jnp.gradient(f_track, t_back)

            T_plunge_s = T_plunge * YEAR_SI
            tau_obs    = T_plunge_s - t_obs_jax
            f_alpha    = jnp.interp(tau_obs, t_back, f_track)
            fdot_alpha = jnp.interp(tau_obs, t_back, fdot_track)

            stat = det_stat(data_sfts, f_alpha, fdot_alpha, T_sft=T_sft)
            return -stat

        batch_fitness = jax.vmap(_single_particle_fitness)
        positions_jax = jnp.array(positions, dtype=jnp.float64)
        fitness = np.array(batch_fitness(positions_jax))
        return fitness

    def _evaluate_fitness(self,
                           positions: np.ndarray,
                           mode: tuple) -> np.ndarray:
        """Evaluate fitness for all particles, using the best available backend."""
        if self.use_fewtrax:
            try:
                return self._batch_fitness_fewtrax(positions, mode)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"fewtrax batch fitness failed ({e}); falling back to FEW backend."
                )
        # Serial FEW fallback
        return np.array([self._fitness_few(p, mode) for p in positions])

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self,
                 mode: tuple = (2, 0),
                 n_particles: int = 200,
                 n_iter: int = 100,
                 T_plunge_estimate: float | None = None,
                 seed: int = 42,
                 w: float = 0.729,
                 c1: float = 1.494,
                 c2: float = 1.494,
                 verbose: bool = True) -> tuple[np.ndarray, float, list[float]]:
        """Run PSO to maximize the semi-coherent detection statistic.

        Parameters
        ----------
        mode : tuple (m, n) or (m, k, n)
            Mode numbers for the frequency track.
        n_particles : int
            Swarm size.
        n_iter : int
            Number of PSO iterations.
        T_plunge_estimate : float, optional
            Estimated plunge time [years] for focused initialization.
            If not given, uniform sampling across the full T_plunge bounds
            is used.
        seed : int
            Random seed.
        w, c1, c2 : float
            PSO hyperparameters (Clerc-Kennedy defaults).
        verbose : bool
            Print global-best update messages.

        Returns
        -------
        best_params : np.ndarray, shape (5,)
            Best ``[M, mu, a, T_plunge, e_f]`` found.
        best_fitness : float
            Negative detection statistic at ``best_params``.
        history : list of float
            Global-best fitness value after each iteration.
        """
        # --- Initialization ---
        if T_plunge_estimate is not None:
            positions, velocities, effective_bounds = initialize_swarm_from_track(
                f0=0.0, fdot0=0.0,
                T_plunge_estimate=T_plunge_estimate,
                bounds=self.bounds,
                n_particles=n_particles,
                seed=seed,
            )
        else:
            scaler = ParameterScaler(self.bounds, seed=seed)
            positions  = scaler.draw_samples(n_particles)
            rng        = np.random.default_rng(seed)
            ranges     = self.bounds[:, 1] - self.bounds[:, 0]
            velocities = rng.uniform(-0.05, 0.05, (n_particles, 5)) * ranges
            effective_bounds = self.bounds

        lower_b = jnp.array(effective_bounds[:, 0])
        upper_b = jnp.array(effective_bounds[:, 1])

        # --- Initial fitness ---
        fitness = self._evaluate_fitness(positions, mode)

        personal_best_pos = positions.copy()
        personal_best_fit = fitness.copy()

        global_best_idx = int(np.argmin(fitness))
        global_best_pos = positions[global_best_idx].copy()
        global_best_fit = float(fitness[global_best_idx])

        history = [global_best_fit]
        key     = jax.random.PRNGKey(seed)

        pos_jax  = jnp.array(positions)
        vel_jax  = jnp.array(velocities)
        pbest_jax = jnp.array(personal_best_pos)
        gbest_jax = jnp.array(global_best_pos)

        if verbose:
            print(f"PSO start: best stat = {-global_best_fit:.4f}")

        # --- Main loop ---
        for iteration in range(n_iter):
            key, subkey = jax.random.split(key)

            pos_jax, vel_jax = pso_update_step(
                pos_jax, vel_jax,
                pbest_jax,
                jnp.array(personal_best_fit),
                gbest_jax,
                subkey,
                w=w, c1=c1, c2=c2,
                lower_bounds=lower_b,
                upper_bounds=upper_b,
                n_dims=5,
            )

            positions_np = np.array(pos_jax)
            fitness = self._evaluate_fitness(positions_np, mode)

            # Update personal bests
            improved = fitness < personal_best_fit
            personal_best_fit = np.where(improved, fitness, personal_best_fit)
            personal_best_pos = np.where(
                improved[:, None], positions_np, personal_best_pos
            )
            pbest_jax = jnp.array(personal_best_pos)

            # Update global best
            best_idx = int(np.argmin(personal_best_fit))
            if personal_best_fit[best_idx] < global_best_fit:
                global_best_fit = float(personal_best_fit[best_idx])
                global_best_pos = personal_best_pos[best_idx].copy()
                gbest_jax = jnp.array(global_best_pos)

                if verbose:
                    print(
                        f"  Iteration {iteration:4d}: "
                        f"new best stat = {-global_best_fit:.4f}  "
                        f"params = {global_best_pos}"
                    )

            history.append(global_best_fit)

        if verbose:
            print(f"PSO done: best stat = {-global_best_fit:.4f}")

        return global_best_pos, global_best_fit, history

    def optimize_with_gradient_refinement(
        self,
        mode: tuple = (2, 0),
        n_particles: int = 200,
        n_iter: int = 100,
        T_plunge_estimate: float | None = None,
        adam_steps: int = 200,
        seed: int = 42,
    ) -> tuple[np.ndarray, float]:
        """PSO global search followed by Adam gradient refinement.

        Requires fewtrax for the gradient step.

        Parameters
        ----------
        mode, n_particles, n_iter, T_plunge_estimate, seed
            Passed to ``optimize()``.
        adam_steps : int
            Number of Adam optimizer steps for local refinement.

        Returns
        -------
        best_params : np.ndarray, shape (5,)
        best_loss : float
            Track-residual loss (not the detection statistic).
        """
        if not self.use_fewtrax or self.flux_data is None:
            raise RuntimeError(
                "gradient refinement requires fewtrax. "
                "Initialize with use_fewtrax=True and provide flux_data."
            )

        from .track_optimizer import TrackOptimizerJAX

        # Derive observed track from the detection-statistic data SFTs
        # (here we only have data_sfts; the caller should provide f_obs/fdot_obs
        # from the STFT search directly to TrackOptimizerJAX)
        raise NotImplementedError(
            "Call TrackOptimizerJAX.optimize_adam() directly with the "
            "PSO best-fit as the starting point."
        )
