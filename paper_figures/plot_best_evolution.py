import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from plot_styles import apply_physrev_style, get_colorblind_palette

# Apply the style
apply_physrev_style()

def load_best_results(h5_file):
    """Load results from HDF5 file"""
    with h5py.File(h5_file, 'r') as f:
        tpl_vector = f['tpl_vector'][()]
        best_losses = f['best_losses'][()]
        best_loss_evolution = f['best_loss_evolution'][()]
        best_fs = f['best_fs'][()]
        param_dict = {k: v[()] for k, v in f['param_dict'].items()}
        snr = f['SNR'][()]
    
    return tpl_vector, best_losses, best_loss_evolution, best_fs, param_dict, snr


if __name__ == "__main__":
    dirs = sorted(glob.glob("../apaper_results/*"))
    # Check if presaved file exists
    presaved_file = 'paper_evolution_data.h5'
    
    if os.path.exists(presaved_file):
        print(f"Loading presaved data from {presaved_file}")
        with h5py.File(presaved_file, 'r') as f:
            tpl_vector = f['tpl_vector'][()]
            all_evolution = f['all_evolution'][()]
            snr = f['snr'][()]
    else:
        print("No presaved data found. Processing directories...")
        tpl_vector = None
        all_evolution = []
        best_fs_list = []
        snr_list = []
        duration_list = []
        
        for output_dir in dirs:
            h5_file = os.path.join(output_dir, 'best_results.h5')
            if not os.path.exists(h5_file):
                print(f"Warning: {h5_file} not found. Skipping.")
                continue

            current_tpl, current_losses, evolution, fs, param_dict, snr_val = load_best_results(h5_file)

            if tpl_vector is None:
                tpl_vector = current_tpl
            elif not np.array_equal(tpl_vector, current_tpl):
                raise ValueError(f"Tpl_vector mismatch in {output_dir}")

            # if '1e-05' in output_dir:
            all_evolution.append(-evolution)
            best_fs_list.append(fs)
            snr_list.append(snr_val)
        
        all_evolution = np.array(all_evolution)
        best_fs = np.array(best_fs_list)
        snr = np.array(snr_list)
        
        # Save the processed data
        with h5py.File(presaved_file, 'w') as f:
            f['tpl_vector'] = tpl_vector
            f['all_evolution'] = all_evolution
            f['snr'] = snr
        print(f"Data saved to {presaved_file}")
    # with h5py.File(h5_file.split('best_results.h5')[0] + 'search_log.h5', 'r') as f:
    #     duration = np.asarray([f['Tmax_1.0'][f'Tpl_ind{ind}']['duration'][()] for ind in range(len(tpl_vector))])
    
    from plot_detection_probability import compute_detection_probability, get_detection_threshold

    # Separate by SNR
    noise_mask = snr < 1
    signal_mask = snr == 30.0

    # Get the maximum evolution across tpl_vector for each iteration
    max_evolution_per_iter = np.max(all_evolution, axis=1)  # Shape: (n_runs, n_iterations)
    
    # iterations
    iterations = np.arange(max_evolution_per_iter.shape[1]) * 25

    
    # Create a new figure for detection probability vs number of templates
    fig2, ax3 = plt.subplots(1, 1, figsize=(6.5, 6.5))

    # Plot detection probability for different SNR groups

    # Define SNRs to analyze
    # snr_values = [1e-5, 20.0, 25.0, 30.0, 35.0, 40.0]
    snr_values = [20.0, 25.0, 30.0, 35.0, 40.0]
    colors = get_colorblind_palette('tol_muted', n_colors=len(snr_values))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*']
    # iterations
    iterations = np.arange(1,max_evolution_per_iter.shape[1]+1) * 25

    for snr_val, color, marker in zip(snr_values, colors, markers):
        signal_mask_current = snr == snr_val
        
        if not np.any(signal_mask_current):
            print(f"Warning: No data found for SNR = {snr_val}")
            continue
            
        # Vectorized computation for all iterations
        detected = []
        for ii in range(len(iterations)):
            all_best_losses_noise = all_evolution[noise_mask, :, ii]
            mean_noise = all_best_losses_noise.mean(axis=0)
            std_noise = all_best_losses_noise.std(axis=0)
            normalized = (all_best_losses_noise - mean_noise) / std_noise
            threshold = get_detection_threshold(normalized, 1e-3)
            
            best_ds = np.max((all_evolution[signal_mask_current, :, ii] - mean_noise) / std_noise, axis=1)
            n_detected = np.sum(best_ds > threshold) / len(best_ds)
            detected.append(n_detected)
        
        detected = np.array(detected)
        n_templates = 512 * iterations * len(tpl_vector)

        ax3.plot(n_templates/1e6, detected, marker=marker, color=color, label=f'SNR = {int(snr_val)}')
        print(f"SNR {snr_val}: Final detection probability = {detected[-1]:.2f}")

    ax3.set_xlabel(r'Total number of templates ($\times 10^6$)')
    ax3.set_ylabel('Detection Probability')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True)
    ax3.legend(ncol=2, loc=(0.3, 0.01))

    # Add second x-axis on top with duration values
    ax3_top = ax3.twiny()
    ax3_top.plot(iterations, n_templates/1e6, alpha=0)  # Invisible plot to set the scale
    ax3_top.set_xlabel('Number of optimization iterations')
    plt.tight_layout()
    plt.savefig('detection_probability_vs_iteration.pdf')

    print(f"Iteration plot saved as 'detection_probability_vs_iteration.pdf'")