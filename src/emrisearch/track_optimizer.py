"""
Gradient-based and multi-start track optimizer for EMRI parameter recovery.

Starting from a single frequency track identified in the STFT basis, this
module provides tools to efficiently recover the 5D physical EMRI parameters
(M, μ, a, T_plunge, e_f).

Two backends are provided:

- **NumPy / FEW backend** (always available): uses ``get_f_fdot_fddot_back``
  from ``emri_utils`` and SciPy Nelder-Mead for gradient-free multi-start
  optimization.  Suitable for the initial coarse scan.

- **JAX / fewtrax backend** (requires ``fewtrax`` in the Python path):
  uses the JAX-differentiable ``EMRIInspiral`` from fewtrax and Adam / L-BFGS
  for gradient-based refinement.  Provides O(50x) fewer function evaluations
  than the gradient-free path.

Usage
-----
::

    from emrisearch.track_optimizer import TrackOptimizer, estimate_plunge_time

    # f_obs, fdot_obs : observed frequency and frequency-derivative per SFT
    # t_obs           : SFT mid-times in seconds
    optimizer = TrackOptimizer(f_obs, fdot_obs, t_obs, T_sft=5e4)

    # Estimate plunge time from the observed track
    T_plunge_est = optimizer.T_plunge_estimate  # years

    # Joint mode-number identification + parameter optimization
    best_mode, best_params, best_loss, all_results = (
        optimizer.identify_mode_and_optimize(n_starts=64)
    )
    # best_params = [M, mu, a, T_plunge, e_f]
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from few.utils.constants import YRSID_SI

from .emri_utils import get_f_fdot_fddot_back
from .draw_population import ParameterScaler

# ---------------------------------------------------------------------------
# Default physical parameter bounds  [M, mu, a, T_plunge, e_f]
# ---------------------------------------------------------------------------

DEFAULT_BOUNDS = np.array([
    [3e5,  5e6],    # M      [M_sun]
    [1.0,  100.0],  # mu     [M_sun]
    [0.0,  0.998],  # a      (dimensionless spin)
    [0.05, 2.0],    # T_plunge  [years]
    [0.0,  0.3],    # e_f    (final eccentricity)
])


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def estimate_plunge_time(f_obs: np.ndarray, fdot_obs: np.ndarray,
                         mode_m: int = 2) -> tuple[float, np.ndarray]:
    """Estimate the time to plunge from the observed frequency track.

    Uses the chirp-time approximation from the Peters formula for a circular
    Kerr orbit near the separatrix::

        T_plunge ≈ f / (|ḟ| · 11/3)

    This is only a rough guide; the exact factor depends on (a, p, e).  The
    estimate is used to narrow the T_plunge search bounds before optimization.

    Parameters
    ----------
    f_obs : array-like, shape (n_sft,)
        Observed GW frequency [Hz] at each SFT segment.
    fdot_obs : array-like, shape (n_sft,)
        Observed frequency derivative [Hz/s].
    mode_m : int
        Dominant azimuthal mode number (default 2).

    Returns
    -------
    T_plunge_estimate : float
        Estimated time to plunge [years] (weighted average over valid segments).
    T_plunge_per_segment : np.ndarray, shape (n_sft,)
        Per-segment estimate [years].
    """
    f_obs = np.asarray(f_obs, dtype=float)
    fdot_obs = np.asarray(fdot_obs, dtype=float)

    valid = (fdot_obs > 0) & (f_obs > 0)
    if not np.any(valid):
        raise ValueError(
            "No valid segments (positive f and fdot) found in the observed track."
        )

    T_plunge_s = f_obs / (fdot_obs * (11.0 / 3.0))  # seconds
    T_plunge_yr = T_plunge_s / YRSID_SI

    # Weight heavier segments (near plunge, where fdot is largest)
    weights = np.where(valid, fdot_obs, 0.0)
    T_est = float(np.average(T_plunge_yr[valid], weights=weights[valid]))
    return T_est, T_plunge_yr


def get_default_mode_candidates(max_m: int = 4, max_n: int = 2) -> list[tuple]:
    """Generate the default set of mode candidates.

    Returns all (m, n) pairs with ``m ∈ [1, max_m]`` and ``n ∈ [-max_n, max_n]``.
    (The polar harmonic k=0 is assumed; most power for equatorial EMRIs.)

    Parameters
    ----------
    max_m : int
        Maximum azimuthal mode number.
    max_n : int
        Maximum radial mode number.

    Returns
    -------
    candidates : list of (m, n) tuples
    """
    candidates = []
    for m in range(1, max_m + 1):
        for n in range(-max_n, max_n + 1):
            candidates.append((m, n))
    return candidates


def compute_track_residuals(params: np.ndarray,
                             mode: tuple,
                             f_obs: np.ndarray,
                             t_obs: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute track residuals between predicted and observed frequencies.

    Uses ``get_f_fdot_fddot_back`` from ``emri_utils`` (NumPy / FEW backend).
    Suitable for the initial coarse scan; does not provide gradients.

    Parameters
    ----------
    params : array-like, shape (5,) or (6,)
        Physical parameters ``[M, mu, a, T_plunge, e_f]`` (x0=1 appended if
        only 5 values given).
    mode : tuple (m, n)
        Mode numbers: frequency = ``m·f_φ + n·f_r``.
    f_obs : array-like, shape (n_sft,)
        Observed GW frequencies [Hz].
    t_obs : array-like, shape (n_sft,)
        Observation times [s].

    Returns
    -------
    residuals : np.ndarray, shape (n_sft,)
        Frequency residuals ``f_pred - f_obs`` [Hz].
    loss : float
        Sum of squared residuals.  Returns ``1e10`` on integration failure.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim == 1 and params.shape[0] == 5:
        params_6d = np.append(params, 1.0)[np.newaxis, :]
    elif params.ndim == 1 and params.shape[0] == 6:
        params_6d = params[np.newaxis, :]
    else:
        params_6d = np.atleast_2d(params)
        if params_6d.shape[1] == 5:
            params_6d = np.column_stack([params_6d, np.ones(params_6d.shape[0])])

    m, n = mode
    f_obs = np.asarray(f_obs, dtype=float)
    t_obs = np.asarray(t_obs, dtype=float)

    try:
        _, f, _, _ = get_f_fdot_fddot_back(params_6d, t_obs)
        f_pred = m * f[0] + n * f[1]
        valid = f_pred > 0
        if not np.any(valid):
            return np.zeros_like(f_obs), 1e10
        residuals = f_pred - f_obs
        residuals[~valid] = 0.0
        loss = float(np.sum(residuals ** 2))
        return residuals, loss
    except Exception:
        return np.zeros_like(f_obs), 1e10


def scan_mode_numbers(params: np.ndarray,
                       mode_candidates: list[tuple],
                       f_obs: np.ndarray,
                       t_obs: np.ndarray) -> tuple[tuple, list[float]]:
    """Rank mode candidates by track residual loss.

    Parameters
    ----------
    params : array-like, shape (5,) or (6,)
        Physical parameters ``[M, mu, a, T_plunge, e_f, (x0)]``.
    mode_candidates : list of (m, n) tuples
        Candidate mode number pairs to evaluate.
    f_obs : array-like, shape (n_sft,)
        Observed frequencies [Hz].
    t_obs : array-like, shape (n_sft,)
        Observation times [s].

    Returns
    -------
    best_mode : tuple (m, n)
        Mode with the smallest residual loss.
    losses : list of float
        Loss value for each candidate.
    """
    losses = []
    for mode in mode_candidates:
        _, loss = compute_track_residuals(params, mode, f_obs, t_obs)
        losses.append(loss)
    best_idx = int(np.argmin(losses))
    return mode_candidates[best_idx], losses


# ---------------------------------------------------------------------------
# TrackOptimizer: multi-start gradient-free optimizer (NumPy / FEW backend)
# ---------------------------------------------------------------------------

class TrackOptimizer:
    """Multi-start optimizer for recovering EMRI parameters from an STFT track.

    Uses the NumPy / FEW backend (``get_f_fdot_fddot_back``) and SciPy
    Nelder-Mead.  No JAX or fewtrax required.

    Parameters
    ----------
    f_obs : array-like, shape (n_sft,)
        Observed GW frequency [Hz] at each SFT segment.
    fdot_obs : array-like, shape (n_sft,)
        Observed frequency derivative [Hz/s].
    t_obs : array-like, shape (n_sft,)
        SFT mid-times [s].
    T_sft : float
        SFT segment duration [s].
    bounds : array-like, optional
        Parameter bounds, shape (5, 2), columns ``[lower, upper]`` for
        ``[M, mu, a, T_plunge, e_f]``.
    T_plunge_width : float
        Half-width [years] around the T_plunge estimate used to narrow
        the search bounds (default 0.3).
    """

    def __init__(self,
                 f_obs: np.ndarray,
                 fdot_obs: np.ndarray,
                 t_obs: np.ndarray,
                 T_sft: float = 5e4,
                 bounds: np.ndarray | None = None,
                 T_plunge_width: float = 0.3):
        self.f_obs = np.asarray(f_obs, dtype=float)
        self.fdot_obs = np.asarray(fdot_obs, dtype=float)
        self.t_obs = np.asarray(t_obs, dtype=float)
        self.T_sft = float(T_sft)

        if bounds is None:
            self.bounds = DEFAULT_BOUNDS.copy()
        else:
            self.bounds = np.asarray(bounds, dtype=float).copy()

        # Estimate plunge time and narrow T_plunge bounds
        self.T_plunge_estimate, self.T_plunge_per_segment = estimate_plunge_time(
            self.f_obs, self.fdot_obs
        )
        self.bounds[3, 0] = max(
            self.bounds[3, 0],
            self.T_plunge_estimate - T_plunge_width,
        )
        self.bounds[3, 1] = min(
            self.bounds[3, 1],
            self.T_plunge_estimate + T_plunge_width,
        )

        self.scaler = ParameterScaler(self.bounds)

    def _objective(self, params_5d: np.ndarray, mode: tuple) -> float:
        """Scalar track-residual loss for Nelder-Mead."""
        _, loss = compute_track_residuals(params_5d, mode, self.f_obs, self.t_obs)
        return loss

    def optimize(self,
                 mode: tuple = (2, 0),
                 n_starts: int = 64,
                 max_iter: int = 200,
                 seed: int = 42) -> tuple[np.ndarray, float, list]:
        """Multi-start Nelder-Mead optimization for a fixed mode.

        Parameters
        ----------
        mode : tuple (m, n)
            Mode numbers to assume.
        n_starts : int
            Number of independent Sobol-sampled starting points.
        max_iter : int
            Maximum iterations per start.
        seed : int
            Random seed for the ParameterScaler.

        Returns
        -------
        best_params : np.ndarray, shape (5,)
            Best ``[M, mu, a, T_plunge, e_f]`` found.
        best_loss : float
            Best loss value.
        all_results : list of dict
            ``{'params': ..., 'loss': ...}`` for every converged start.
        """
        # Use a fresh scaler with the current seed
        scaler = ParameterScaler(self.bounds, seed=seed)
        init_params = scaler.draw_samples(n_starts)

        best_loss = np.inf
        best_params = init_params[0].copy()
        all_results = []

        for params_init in init_params:
            try:
                result = minimize(
                    self._objective,
                    params_init,
                    args=(mode,),
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-15},
                )
                params_clipped = np.clip(result.x, self.bounds[:, 0], self.bounds[:, 1])
                loss = self._objective(params_clipped, mode)
                all_results.append({'params': params_clipped.copy(), 'loss': loss})

                if loss < best_loss:
                    best_loss = loss
                    best_params = params_clipped.copy()
            except Exception:
                continue

        return best_params, best_loss, all_results

    def identify_mode_and_optimize(self,
                                    mode_candidates: list[tuple] | None = None,
                                    n_starts: int = 64,
                                    max_iter: int = 200) -> tuple:
        """Joint mode identification and parameter optimization.

        Runs ``optimize()`` for each mode candidate and returns the
        combination with the smallest track-residual loss.

        Parameters
        ----------
        mode_candidates : list of (m, n) tuples, optional
            Modes to scan.  Defaults to m ∈ [1,4], n ∈ [-2,2].
        n_starts : int
            Number of random starts per mode.
        max_iter : int
            Nelder-Mead iteration limit per start.

        Returns
        -------
        best_mode : tuple (m, n)
        best_params : np.ndarray, shape (5,)
        best_loss : float
        results_by_mode : dict
            Keyed by mode tuple; values are
            ``{'params': ..., 'loss': ..., 'all_results': ...}``.
        """
        if mode_candidates is None:
            mode_candidates = get_default_mode_candidates(max_m=4, max_n=2)

        results_by_mode = {}
        best_loss = np.inf
        best_mode = mode_candidates[0]
        best_params = None

        for mode in mode_candidates:
            params, loss, all_res = self.optimize(
                mode=mode, n_starts=n_starts, max_iter=max_iter
            )
            results_by_mode[mode] = {
                'params': params,
                'loss': loss,
                'all_results': all_res,
            }
            if loss < best_loss:
                best_loss = loss
                best_mode = mode
                best_params = params

        return best_mode, best_params, best_loss, results_by_mode


# ---------------------------------------------------------------------------
# TrackOptimizerJAX: gradient-based optimizer using fewtrax (optional)
# ---------------------------------------------------------------------------

class TrackOptimizerJAX:
    """Gradient-based EMRI parameter optimizer using the fewtrax backend.

    Requires ``fewtrax`` to be installed::

        pip install /path/to/JAX-waveform/fewtrax

    This class provides:

    - A differentiable track-residual loss ``L(θ)`` via ``fewtrax.EMRIInspiral``
      backward integration and JAX automatic differentiation.
    - Multi-start Adam optimization (via ``optax``) for robust convergence.
    - Optional L-BFGS refinement (via ``jaxopt``) for high-precision results.

    Parameters
    ----------
    f_obs : array-like, shape (n_sft,)
        Observed GW frequency [Hz].
    fdot_obs : array-like, shape (n_sft,)
        Observed frequency derivative [Hz/s].
    t_obs : array-like, shape (n_sft,)
        SFT mid-times [s].
    flux_data : fewtrax.data.FluxData
        Pre-loaded fewtrax flux data.
    T_sft : float
        SFT segment duration [s].
    bounds : array-like, optional
        Parameter bounds, shape (5, 2).
    T_plunge_width : float
        Half-width [years] for T_plunge bound narrowing.
    """

    def __init__(self,
                 f_obs: np.ndarray,
                 fdot_obs: np.ndarray,
                 t_obs: np.ndarray,
                 flux_data,
                 T_sft: float = 5e4,
                 bounds: np.ndarray | None = None,
                 T_plunge_width: float = 0.3):
        try:
            import jax
            import jax.numpy as jnp
            import optax
            from fewtrax.trajectory import EMRIInspiral
            from fewtrax.utils.geodesic import get_fundamental_frequencies
            from fewtrax.utils.constants import YEAR_SI, MTSUN_SI
        except ImportError as exc:
            raise ImportError(
                "TrackOptimizerJAX requires fewtrax, jax, and optax. "
                "Install fewtrax from the JAX-waveform repository."
            ) from exc

        import jax
        import jax.numpy as jnp
        import optax

        self._jax = jax
        self._jnp = jnp
        self._optax = optax

        self.f_obs = jnp.array(f_obs, dtype=jnp.float64)
        self.fdot_obs = np.asarray(fdot_obs, dtype=float)
        self.t_obs = jnp.array(t_obs, dtype=jnp.float64)
        self.T_sft = float(T_sft)
        self.flux_data = flux_data

        if bounds is None:
            self.bounds = DEFAULT_BOUNDS.copy()
        else:
            self.bounds = np.asarray(bounds, dtype=float).copy()

        self.T_plunge_estimate, self.T_plunge_per_segment = estimate_plunge_time(
            np.asarray(f_obs), np.asarray(fdot_obs)
        )
        self.bounds[3, 0] = max(
            self.bounds[3, 0], self.T_plunge_estimate - T_plunge_width
        )
        self.bounds[3, 1] = min(
            self.bounds[3, 1], self.T_plunge_estimate + T_plunge_width
        )
        self.scaler = ParameterScaler(self.bounds)

    def _build_loss(self, mode: tuple):
        """Return a JAX-compilable loss function for the given mode (m, k, n)."""
        jax = self._jax
        jnp = self._jnp

        from fewtrax.trajectory import EMRIInspiral
        from fewtrax.utils.geodesic import get_fundamental_frequencies
        from fewtrax.utils.constants import YEAR_SI, MTSUN_SI

        traj = EMRIInspiral(self.flux_data)
        f_obs = self.f_obs
        t_obs = self.t_obs
        m, k, n = mode if len(mode) == 3 else (mode[0], 0, mode[1])

        @jax.jit
        def loss(theta):
            """theta = [M, mu, a, T_plunge, e_f], all JAX floats."""
            M, mu, a, T_plunge, e_f = theta[0], theta[1], theta[2], theta[3], theta[4]

            # Backward integration from separatrix
            t_back, p_back, e_back, _, _, _ = traj(
                p0=jnp.float64(10.0),   # ignored in backward mode
                e0=e_f,
                T=T_plunge,
                a=a,
                M=M,
                mu=mu,
                backward=True,
                e_f=e_f,
                dense_steps=200,
            )

            # Frequency track along the trajectory
            M_total_s = (M + mu) * MTSUN_SI

            def freq_one(p, e):
                Om_phi, Om_theta, Om_r = get_fundamental_frequencies(
                    jnp.abs(a), p, e, 1.0
                )
                return jnp.abs(m * Om_phi + k * Om_theta + n * Om_r) / (
                    2.0 * jnp.pi * M_total_s
                )

            f_track = jax.vmap(freq_one)(p_back, e_back)

            # Map SFT observation times to τ = T_plunge - t
            T_plunge_s = T_plunge * YEAR_SI
            tau_obs = T_plunge_s - t_obs

            # Interpolate predicted track to observation τ values
            f_pred = jnp.interp(tau_obs, t_back, f_track)

            return jnp.sum((f_pred - f_obs) ** 2)

        return loss

    def optimize_adam(self,
                      mode: tuple = (2, 0),
                      n_starts: int = 32,
                      n_steps: int = 300,
                      learning_rate: float = 1e-3,
                      seed: int = 42) -> tuple[np.ndarray, float]:
        """Multi-start Adam optimization.

        Parameters
        ----------
        mode : tuple (m, n) or (m, k, n)
            Mode numbers (k=0 assumed for (m, n) tuples).
        n_starts : int
            Number of Sobol-sampled starting points.
        n_steps : int
            Adam steps per start.
        learning_rate : float
            Adam learning rate.
        seed : int
            Random seed for ParameterScaler.

        Returns
        -------
        best_params : np.ndarray, shape (5,)
            Best ``[M, mu, a, T_plunge, e_f]``.
        best_loss : float
            Best track-residual loss.
        """
        import jax
        import jax.numpy as jnp
        import optax

        loss_fn = self._build_loss(mode)
        optimizer = optax.adam(learning_rate=learning_rate)

        @jax.jit
        def step(theta, opt_state):
            loss_val, grads = jax.value_and_grad(loss_fn)(theta)
            updates, new_state = optimizer.update(grads, opt_state)
            new_theta = optax.apply_updates(theta, updates)
            # Clip to bounds
            lower = jnp.array(self.bounds[:, 0])
            upper = jnp.array(self.bounds[:, 1])
            new_theta = jnp.clip(new_theta, lower, upper)
            return new_theta, new_state, loss_val

        scaler = ParameterScaler(self.bounds, seed=seed)
        init_params = scaler.draw_samples(n_starts)

        best_loss = np.inf
        best_params = init_params[0].copy()

        for params_init in init_params:
            theta = jnp.array(params_init, dtype=jnp.float64)
            opt_state = optimizer.init(theta)

            for _ in range(n_steps):
                theta, opt_state, loss_val = step(theta, opt_state)

            final_loss = float(loss_fn(theta))
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = np.array(theta)

        return best_params, best_loss

    def optimize_lbfgs(self,
                        params_init: np.ndarray,
                        mode: tuple = (2, 0),
                        maxiter: int = 200) -> tuple[np.ndarray, float]:
        """L-BFGS refinement from a given starting point.

        Requires ``jaxopt``.

        Parameters
        ----------
        params_init : array-like, shape (5,)
            Starting parameters ``[M, mu, a, T_plunge, e_f]``.
        mode : tuple (m, n) or (m, k, n)
        maxiter : int

        Returns
        -------
        best_params : np.ndarray, shape (5,)
        best_loss : float
        """
        try:
            import jaxopt
        except ImportError as exc:
            raise ImportError(
                "L-BFGS refinement requires jaxopt: pip install jaxopt"
            ) from exc

        import jax.numpy as jnp

        loss_fn = self._build_loss(mode)
        lbfgs = jaxopt.LBFGS(fun=loss_fn, maxiter=maxiter)
        result = lbfgs.run(jnp.array(params_init, dtype=jnp.float64))
        best_params = np.clip(
            np.array(result.params),
            self.bounds[:, 0], self.bounds[:, 1]
        )
        best_loss = float(loss_fn(result.params))
        return best_params, best_loss

    def optimize_full(self,
                       mode: tuple = (2, 0),
                       n_starts: int = 32,
                       adam_steps: int = 300,
                       lbfgs_iter: int = 200) -> tuple[np.ndarray, float]:
        """Full pipeline: multi-start Adam followed by L-BFGS refinement.

        Parameters
        ----------
        mode : tuple
        n_starts : int
        adam_steps : int
        lbfgs_iter : int

        Returns
        -------
        best_params : np.ndarray, shape (5,)
        best_loss : float
        """
        adam_params, adam_loss = self.optimize_adam(
            mode=mode, n_starts=n_starts, n_steps=adam_steps
        )
        try:
            best_params, best_loss = self.optimize_lbfgs(
                adam_params, mode=mode, maxiter=lbfgs_iter
            )
        except ImportError:
            best_params, best_loss = adam_params, adam_loss

        return best_params, best_loss

    def compute_fisher(self,
                        params: np.ndarray,
                        mode: tuple = (2, 0),
                        sigma_f: float | None = None) -> np.ndarray:
        """Compute the Fisher information matrix at the given parameters.

        Uses the Jacobian of the predicted frequency track w.r.t. θ::

            F_ij = J^T J / σ_f²   where  J_ai = ∂f_pred(t_a) / ∂θ_i

        Parameters
        ----------
        params : array-like, shape (5,)
            ``[M, mu, a, T_plunge, e_f]``.
        mode : tuple
        sigma_f : float, optional
            Frequency uncertainty [Hz].  Defaults to ``1 / T_sft``.

        Returns
        -------
        F : np.ndarray, shape (5, 5)
            Fisher information matrix.
        """
        import jax
        import jax.numpy as jnp

        if sigma_f is None:
            sigma_f = 1.0 / self.T_sft

        from fewtrax.trajectory import EMRIInspiral
        from fewtrax.utils.geodesic import get_fundamental_frequencies
        from fewtrax.utils.constants import YEAR_SI, MTSUN_SI

        traj = EMRIInspiral(self.flux_data)
        t_obs = self.t_obs
        m, k, n = mode if len(mode) == 3 else (mode[0], 0, mode[1])

        @jax.jit
        def predict_freqs(theta):
            M, mu, a, T_plunge, e_f = (
                theta[0], theta[1], theta[2], theta[3], theta[4]
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

            f_track = jax.vmap(freq_one)(p_back, e_back)
            T_plunge_s = T_plunge * YEAR_SI
            tau_obs = T_plunge_s - t_obs
            return jnp.interp(tau_obs, t_back, f_track)

        theta_jax = jnp.array(params, dtype=jnp.float64)
        J = jax.jacobian(predict_freqs)(theta_jax)  # shape (n_sft, 5)
        J_np = np.array(J)
        F = (J_np.T @ J_np) / sigma_f ** 2
        return F
