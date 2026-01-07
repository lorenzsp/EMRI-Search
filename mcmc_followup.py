#!/usr/bin/env python3
"""
MCMC Follow-up Script for EMRI Parameter Estimation

This script performs MCMC parameter estimation on simulated EMRI signals using
the eryn sampler. It generates an injection, computes the detection statistic
likelihood, and samples the posterior distribution.

Usage:
    python mcmc_followup.py --niterations 1000 --nwalkers 16 --ntemps 4 --nprocesses 4

Output:
    - mcmc_results.h5: MCMC chain backend
    - mcmc_summary.npz: Summary of posterior samples
    - corner_plot.png: Corner plot of posterior
    - loglike_chain.png: Log-likelihood evolution
"""

import argparse
import os
import numpy as np
import corner
import matplotlib.pyplot as plt
from da_utils import sft_inner_product, psd
from de_proposal import DEMove
from eryn.ensemble import EnsembleSampler
from eryn.moves import GaussianMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend
from multiprocess import Pool
from few.utils.constants import YRSID_SI

from emri_utils import get_f_fdot_fddot_back
from search_utils import generate_emri_signal_and_sfts, det_stat


# =============================================================================
# Configuration
# =============================================================================

# SFT and observation parameters
T_SFT = 5e4          # SFT duration [seconds]
T_DATA = 1.0         # Total observation time [years]
DELTA_T = 5.0        # Time step [seconds]
SNR_REF = 30.0       # Reference SNR for injection
T_SNR = 1.0          # SNR reference duration [years]

# Default injection parameters (same as QuickStartEMRIsearch notebook)
DEFAULT_INJECTION = {
    'm1': 1e6,       # Primary mass [solar masses]
    'm2': 10.0,      # Secondary mass [solar masses]
    'a': 0.9,        # Spin parameter
    'Tpl': 0.9,      # Plunge time [years]
    'ef': 0.1,      # Final eccentricity
}

# Parameter bounds for MCMC (using log10(m1) for better sampling)
PARAM_BOUNDS = np.array([
    (np.log10(5e5), np.log10(5e6)),  # log10(m1) [log10(Msun)]
    (1.0, 100.0),                     # m2 [Msun]
    (0.0, 0.998),                     # a (spin)
    (0.8, 1.0),                       # Tpl [years]
    (0.0, 0.2),                       # ef (eccentricity)
])

PARAM_NAMES = ['log10_m1', 'm2', 'a', 'Tpl', 'ef']
NDIM = len(PARAM_NAMES)


# =============================================================================
# Data Generation
# =============================================================================

def generate_injection_data(injection_params, seed=42):
    """
    Generate simulated EMRI signal and SFT data.
    
    Parameters
    ----------
    injection_params : dict
        Dictionary with keys: m1, m2, a, Tpl, ef
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - data_sfts: SFT data (signal + noise)
        - noise_sfts: Pure noise SFTs
        - signal_sfts: Pure signal SFTs
        - t_alpha: Time array
        - true_params: Array of true parameters [log10(m1), m2, a, Tpl, ef]
    """
    np.random.seed(seed)
    
    # Pack parameters for signal generation
    true_values = np.array([
        injection_params['m1'],
        injection_params['m2'],
        injection_params['a'],
        injection_params['Tpl'],
        injection_params['ef'],
        1.0  # x0 scaling factor
    ])
    
    # Generate signal and SFTs
    injection_dict = generate_emri_signal_and_sfts(
        true_values, T_DATA, T_SFT, DELTA_T, SNR_REF, T_SNR
    )
    
    # Extract relevant data
    t_alpha = injection_dict['t_alpha']
    data_sfts = np.asarray(injection_dict['data_sfts'], dtype=np.complex128)
    noise_sfts = np.asarray(injection_dict['noise_sfts'], dtype=np.complex128)
    signal_sfts = np.asarray(injection_dict['signal_sfts'], dtype=np.complex128)
    
    # True parameters in MCMC parameterization (log10(m1))
    true_params = np.array([
        np.log10(injection_params['m1']),
        injection_params['m2'],
        injection_params['a'],
        injection_params['Tpl'],
        injection_params['ef'],
    ])
    
    return {
        'data_sfts': data_sfts,
        'noise_sfts': noise_sfts,
        'signal_sfts': signal_sfts,
        't_alpha': t_alpha,
        'true_params': true_params,
        'injection_dict': injection_dict,
    }


# =============================================================================
# Likelihood and Priors
# =============================================================================

def compute_log_likelihood(params, t_alpha, data_sfts, m=2, n=0, delta_phi_max=1.0):
    """
    Compute log-likelihood for EMRI detection statistic.
    
    Parameters
    ----------
    params : array-like
        Parameter array [log10(m1), m2, a, Tpl, ef]
    t_alpha : array
        Time array [seconds]
    data_sfts : array
        Complex SFT data
    m, n : int
        Harmonic indices
    delta_phi_max : float
        Maximum phase deviation threshold
        
    Returns
    -------
    float
        Log-likelihood value
    """
    # Convert parameters to physical values
    physical_params = np.array([
        10**params[0],  # m1 from log10(m1)
        params[1],      # m2
        params[2],      # a (spin)
        params[3],      # Tpl
        params[4],      # ef
        1.0             # x0
    ])
    
    try:
        phi, f, dotf, dotdotf = get_f_fdot_fddot_back(physical_params, t_alpha, err=1e-6)
    except Exception:
        return -1e50  # Return very low likelihood for failed waveform generation
    
    # Compute frequency track for harmonic (m, n)
    phi_alpha = m * phi[0] + n * phi[1]
    f_alpha = m * f[0] + n * f[1]
    fdot_alpha = m * dotf[0] + n * dotf[1]
    fddot_alpha = m * dotdotf[0] + n * dotdotf[1]
    
    # Phase coherence condition
    delta_phi_approx = np.abs(fddot_alpha * T_SFT**3 / 6)
    
    # Amplitude mask: valid frequency bins
    A_alpha = np.where(
        (f_alpha > 10**(-3.5)) 
        # & 
        # (fdot_alpha > 1e-13) 
        # & 
        # (delta_phi_approx < delta_phi_max)
        ,
        1.0, 0.0
    )
    
    # Compute detection statistic
    dh, hh = det_stat(data_sfts, A_alpha, phi_alpha, f_alpha, fdot_alpha, T_sft=T_SFT)
    
    # Sum detection statistic over valid bins
    log_like = 0.5 * np.sum(A_alpha * np.divide(
        np.abs(dh)**2, hh, where=hh > 1e-20, out=np.zeros_like(hh)
    ))
    
    return log_like


def create_priors(bounds):
    """
    Create prior distributions for MCMC sampling.
    
    Parameters
    ----------
    bounds : array
        Parameter bounds of shape (ndim, 2)
        
    Returns
    -------
    dict
        Dictionary with eryn prior container
    """
    priors = {
        "emri": ProbDistContainer({
            i: uniform_dist(bounds[i, 0], bounds[i, 1])
            for i in range(len(bounds))
        })
    }
    return priors


# =============================================================================
# MCMC Initialization
# =============================================================================

def initialize_walkers(true_params, bounds, nwalkers, ntemps, scatter_scale=1e-4):
    """
    Initialize MCMC walkers around the true injection parameters.
    
    Parameters
    ----------
    true_params : array
        True parameter values
    bounds : array
        Parameter bounds
    nwalkers : int
        Number of walkers per temperature
    ntemps : int
        Number of temperatures
    scatter_scale : float
        Scale of Gaussian scatter around true values (fraction of prior width)
        
    Returns
    -------
    dict
        Initial coordinates for eryn sampler
    """
    ndim = len(true_params)
    total_walkers = nwalkers * ntemps
    
    # Compute scatter width based on prior range
    prior_width = bounds[:, 1] - bounds[:, 0]
    scatter_std = scatter_scale * prior_width
    
    # Generate initial positions with Gaussian scatter around truth
    initial_positions = np.zeros((total_walkers, ndim))
    for i in range(total_walkers):
        position = true_params + np.random.randn(ndim) * scatter_std
        # Enforce bounds
        position = np.clip(position, bounds[:, 0] + 1e-6, bounds[:, 1] - 1e-6)
        initial_positions[i] = position
    
    # Reshape for eryn: (ntemps, nwalkers, nbranches, ndim)
    coords = {
        "emri": initial_positions.reshape((ntemps, nwalkers, 1, ndim))
    }
    
    return coords, initial_positions


def compute_initial_covariance(initial_positions):
    """
    Compute covariance matrix from initial walker positions.
    
    Parameters
    ----------
    initial_positions : array
        Initial walker positions of shape (n_samples, ndim)
        
    Returns
    -------
    array
        Covariance matrix
    """
    return np.cov(initial_positions, rowvar=False)


# =============================================================================
# MCMC Callback and Diagnostics
# =============================================================================

def create_update_callback(true_params, true_ll, output_dir, nwalkers, ntemps, ndim, t_alpha):
    """
    Create callback function for MCMC progress updates.
    
    Parameters
    ----------
    true_params : array
        True injection parameters
    true_ll : float
        True log-likelihood value
    output_dir : str
        Directory for output plots
    nwalkers, ntemps, ndim : int
        Sampler dimensions
    t_alpha : array
        Time array for computing frequency tracks
        
    Returns
    -------
    callable
        Callback function for eryn sampler
    """
    # Add nan for corner plot compatibility (extra dimension for ll)
    truths_with_ll = np.append(true_params, np.nan)
    
    # Precompute true frequency track (m=2, n=0 harmonic)
    true_physical_params = np.array([
        10**true_params[0], true_params[1], true_params[2],
        true_params[3], true_params[4], 1.0
    ])
    phi_true, f_true, _, _ = get_f_fdot_fddot_back(true_physical_params, t_alpha)
    f_alpha_true = 2 * f_true[0]  # m=2, n=0
    
    def update_callback(iteration, result, sampler):
        """Callback executed periodically during MCMC."""
        print('\n' + '='*60)
        print(f'MCMC Progress Update - Iteration {iteration}')
        print('='*60)
        
        # Get current state
        last_state = sampler.get_last_sample()
        last_ll = last_state.log_like
        all_coords = last_state.branches_coords["emri"][:, :, 0, :].reshape(-1, ndim)
        
        # Find best walker
        best_idx = np.argmax(last_ll)
        best_params = all_coords[best_idx]
        best_ll = np.max(last_ll)
        
        print(f'Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}')
        print(f'Best log-likelihood: {best_ll:.2f} (True: {true_ll:.2f})')
        print(f'Best parameters: {best_params}')
        print(f'True parameters: {true_params}')
        
        # Generate diagnostic plots for cold chain
        discard = max(1, int(sampler.iteration * 0.1))
        cold_chain = sampler.get_chain(discard=discard, thin=1)["emri"][:, 0].reshape(-1, ndim)
        cold_ll = sampler.get_log_like(discard=discard, thin=1)[:, 0].flatten()
        
        # Corner plot
        samples_with_ll = np.column_stack([cold_chain, cold_ll])
        labels = PARAM_NAMES + ['log_like']
        fig = corner.corner(
            samples_with_ll,
            labels=labels,
            truths=truths_with_ll,
            levels=1 - np.exp(-0.5 * np.array([1, 2, 3])**2),
            show_titles=True,
        )
        fig.savefig(os.path.join(output_dir, 'corner_plot.png'), dpi=150)
        plt.close(fig)
        
        # Log-likelihood evolution
        full_ll_chain = sampler.get_log_like(discard=0, thin=1)[:, 0]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(full_ll_chain, alpha=0.7)
        ax.axhline(true_ll, color='r', linestyle='--', label=f'True LL: {true_ll:.1f}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-Likelihood')
        ax.set_title('Log-Likelihood Evolution (Cold Chain)')
        ax.legend()
        fig.savefig(os.path.join(output_dir, 'loglike_chain.png'), dpi=150)
        plt.close(fig)
        
        # Update proposal distributions adaptively
        if sampler.iteration > 100:
            cov = np.cov(cold_chain, rowvar=False) / ndim
            temp_move = GaussianMove({"emri": cov}, factor=10)
            sampler.moves[1].all_proposal = temp_move.all_proposal
        
        # ---- Frequency track comparison plot (2 subplots) ----
        # Get current walker positions at temperature 0 (last iteration only)
        temp0_coords = last_state.branches_coords["emri"][0, :, 0, :]  # shape: (nwalkers, ndim)
        temp0_ll = last_ll[0, :]  # Log-likelihoods for temp 0 walkers (last iteration)
        
        fig_freq, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Plot true frequency track
        t_days = t_alpha / 86400  # Convert to days
        ax_top.plot(t_days, f_alpha_true, 'k-', linewidth=2, label='True', zorder=10)
        
        # Compute and plot f_alpha for each walker at temperature 0 (last iteration)
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=temp0_ll.min(), vmax=temp0_ll.max())
        
        for i, walker_params in enumerate(temp0_coords):
            try:
                physical_params = np.array([
                    10**walker_params[0], walker_params[1], walker_params[2],
                    walker_params[3], walker_params[4], 1.0
                ])
                _, f_walker, _, _ = get_f_fdot_fddot_back(physical_params, t_alpha)
                f_alpha_walker = 2 * f_walker[0]  # m=2, n=0
                
                color = cmap(norm(temp0_ll[i]))
                ax_top.plot(t_days, f_alpha_walker, '-', color=color, alpha=0.5, linewidth=1)
                
                # Relative difference
                rel_diff = np.abs(f_alpha_walker - f_alpha_true) / f_alpha_true
                ax_bottom.semilogy(t_days, rel_diff, '-', color=color, alpha=0.5, linewidth=1)
            except Exception:
                continue  # Skip walkers that fail waveform generation
        
        # Top subplot formatting
        ax_top.set_ylabel('Frequency $f_\\alpha$ [Hz]')
        ax_top.legend(loc='upper left')
        ax_top.set_title(f'Frequency Tracks - Iteration {sampler.iteration}')
        ax_top.grid(True, alpha=0.3)
        
        # Bottom subplot formatting
        ax_bottom.axhline(0, color='k', linestyle='--', linewidth=1)
        ax_bottom.set_xlabel('Time [days]')
        ax_bottom.set_ylabel('Relative Difference $(f - f_{true})/f_{true}$')
        ax_bottom.grid(True, alpha=0.3)
        
        # Add colorbar horizontally above the top plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig_freq.add_axes([0.15, 0.92, 0.7, 0.02])  # [left, bottom, width, height]
        cbar = fig_freq.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Log-Likelihood')
        cbar_ax.xaxis.set_ticks_position('top')
        cbar_ax.xaxis.set_label_position('top')
        
        plt.tight_layout(rect=[0, 0, 1, 0.90])  # Leave space at top for colorbar
        fig_freq.savefig(os.path.join(output_dir, 'frequency_tracks.png'), dpi=150)
        plt.close(fig_freq)
        
        print('='*60 + '\n')
    
    return update_callback


# =============================================================================
# Main Execution
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MCMC parameter estimation for EMRI signals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--niterations', type=int, default=1000,
                        help='Number of MCMC iterations')
    parser.add_argument('--nwalkers', type=int, default=8,
                        help='Number of walkers per temperature')
    parser.add_argument('--ntemps', type=int, default=3,
                        help='Number of parallel temperatures')
    parser.add_argument('--nprocesses', type=int, default=4,
                        help='Number of parallel processes')
    parser.add_argument('--output-dir', type=str, default='./mcmc_results',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=2601,
                        help='Random seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing backend')
    return parser.parse_args()


if __name__ == '__main__':
    """Main execution function."""
    args = parse_arguments()
    
    # Setup
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    backend_path = os.path.join(args.output_dir, 'mcmc_results.h5')
    
    print('='*60)
    print('EMRI MCMC Parameter Estimation')
    print('='*60)
    print(f'Iterations: {args.niterations}')
    print(f'Walkers: {args.nwalkers}')
    print(f'Temperatures: {args.ntemps}')
    print(f'Processes: {args.nprocesses}')
    print(f'Output: {args.output_dir}')
    print('='*60)
    
    # -------------------------------------------------------------------------
    # Step 1: Generate injection data
    # -------------------------------------------------------------------------
    print('\n[1/4] Generating injection data...')
    injection_data = generate_injection_data(DEFAULT_INJECTION, seed=args.seed)
    # data with noise
    data_sfts = injection_data['data_sfts']
    # data without noise
    # data_sfts = injection_data['signal_sfts']
    t_alpha = injection_data['t_alpha']
    true_params = injection_data['true_params']
    
    print(f'  Data shape: {data_sfts.shape}')
    print(f'  Time array: {len(t_alpha)} points')
    print(f'  True params: {true_params}')
    
    # Prepare likelihood arguments (use t_alpha[:-1] to match SFT count)
    num_sfts = data_sfts.shape[1]
    like_args = (t_alpha[:num_sfts], data_sfts)
    print(f"  Data loaded. Number of SFTs: {like_args[0].shape}, {like_args[1].shape}")
    
    # Compute true log-likelihood
    true_ll = compute_log_likelihood(true_params, *like_args)
    print(f'  True log-likelihood: {true_ll:.2f}')
    print(f'  sqrt(2 x log-likelihood): {(2*true_ll)**0.5:.2f}')
    
    # Compute matched SNR for reference
    samples_per_sft = int(T_SFT/DELTA_T) 
    frequencies = np.fft.rfftfreq(samples_per_sft, DELTA_T)
    psd_f = psd(frequencies)
    matched_snr = np.sqrt(sft_inner_product(injection_data['signal_sfts'], injection_data['data_sfts'], frequencies).sum())
    print(f'  Matched SNR of injection: {matched_snr:.2f}')

    # plot data_sfts and 
    plt.figure(figsize=(8/1.5,4.5/1.5))
    plt.imshow(np.abs(data_sfts/psd_f[:, None]**0.5),aspect='auto',origin='lower',extent=[0, t_alpha[-1]/86400, frequencies[0], frequencies[-1]],cmap='viridis')
    plt.axhline(10**(-3.5),color='w',linestyle='--',label='f = 10^-3.5 Hz')
    plt.ylim(1e-4, 0.05)
    plt.legend(loc='upper left')
    plt.xlabel('Time [days]')
    plt.ylabel('Frequency [Hz]')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'data_sfts.png'), dpi=150)
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize MCMC
    # -------------------------------------------------------------------------
    print('\n[2/4] Initializing MCMC...')
    
    # Create priors
    priors = create_priors(PARAM_BOUNDS)
    
    # Initialize walkers around true parameters
    if args.resume and os.path.exists(backend_path):
        print('  Resuming from existing backend...')
        backend = HDFBackend(backend_path)
        chain = backend.get_chain()["emri"]
        initial_positions = chain[-1, 0].reshape(-1, NDIM)
        coords = {"emri": initial_positions.reshape(
            (args.ntemps, args.nwalkers, 1, NDIM)
        )}
        covariance = np.cov(chain[-100:, 0].reshape(-1, NDIM), rowvar=False)
    else:
        # Fresh start: initialize around true values
        coords, initial_positions = initialize_walkers(
            true_params, PARAM_BOUNDS, args.nwalkers, args.ntemps,
            scatter_scale=1e-4
        )
        covariance = compute_initial_covariance(initial_positions)
        # Remove existing backend if not resuming
        if os.path.exists(backend_path):
            os.remove(backend_path)
    
    print(f'  Initial positions shape: {initial_positions.shape}')
    
    # Setup moves: combination of differential evolution and Gaussian proposals
    moves = [
        DEMove(F=0.5, CR=1.0, use_current_state=True),
        GaussianMove({"emri": covariance / NDIM}, factor=10),
    ]
    
    # Create callback for progress updates
    update_callback = create_update_callback(
        true_params, true_ll, args.output_dir,
        args.nwalkers, args.ntemps, NDIM, like_args[0]
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Run MCMC
    # -------------------------------------------------------------------------
    print('\n[3/4] Running MCMC...')
    
    tempering_kwargs = {'ntemps': args.ntemps}
    
    with Pool(processes=args.nprocesses) as pool:
        sampler = EnsembleSampler(
            args.nwalkers,
            {"emri": NDIM},
            lambda x: compute_log_likelihood(x, *like_args),
            priors,
            branch_names=["emri"],
            nbranches=1,
            backend=backend_path,
            tempering_kwargs=tempering_kwargs,
            moves=moves,
            pool=pool,
            update_fn=update_callback,
            update_iterations=50,
        )
        
        sampler.run_mcmc(coords, args.niterations, burn=0, progress=True, thin_by=1)
    
    print(f'\nMCMC complete. Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}')
    
    # -------------------------------------------------------------------------
    # Step 4: Save results
    # -------------------------------------------------------------------------
    print('\n[4/4] Saving results...')
    
    # Extract posterior samples (cold chain, discarding burn-in)
    discard = args.niterations // 2
    posterior_samples = sampler.get_chain(discard=discard, thin=1)["emri"][:, 0].reshape(-1, NDIM)
    posterior_ll = sampler.get_log_like(discard=discard, thin=1)[:, 0].flatten()
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'mcmc_summary.npz')
    np.savez(
        summary_path,
        samples=posterior_samples,
        log_like=posterior_ll,
        true_params=true_params,
        true_ll=true_ll,
        param_names=PARAM_NAMES,
    )
    print(f'  Saved summary to {summary_path}')
    
    # Final corner plot
    samples_with_ll = np.column_stack([posterior_samples, posterior_ll])
    truths_with_ll = np.append(true_params, np.nan)
    labels = PARAM_NAMES + ['log_like']
    
    fig = corner.corner(
        samples_with_ll,
        labels=labels,
        truths=truths_with_ll,
        levels=1 - np.exp(-0.5 * np.array([1, 2, 3])**2),
        show_titles=True,
        title_fmt='.4f',
    )
    fig.savefig(os.path.join(args.output_dir, 'corner_plot_final.png'), dpi=200)
    plt.close(fig)
    
    print('\n' + '='*60)
    print('MCMC Parameter Estimation Complete')
    print('='*60)
    print(f'Results saved to: {args.output_dir}')
    print(f'Posterior samples: {posterior_samples.shape}')
    
    # Print parameter summary
    print('\nParameter estimates (median ± 1σ):')
    for i, name in enumerate(PARAM_NAMES):
        median = np.median(posterior_samples[:, i])
        low, high = np.percentile(posterior_samples[:, i], [16, 84])
        true_val = true_params[i]
        print(f'  {name}: {median:.4f} [{low:.4f}, {high:.4f}] (true: {true_val:.4f})')
