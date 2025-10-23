# singularity exec --nv sandbox-jax python 
import h5py
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import glob
from scipy.stats import gumbel_r
from plot_styles import apply_physrev_style

# Apply the style
apply_physrev_style()

def load_best_results(h5_file):
    
    with h5py.File(h5_file, 'r') as f:
        tpl_vector = f['tpl_vector'][()]
        best_losses = f['best_losses'][()]
        # best_loss_evolution = f['best_loss_evolution'][()]
        # if ~np.all(np.abs(1-best_losses /best_loss_evolution[:,-1])<1e-2):
        #     print(np.abs(1-best_losses /best_loss_evolution[:,-1]))
        #     print("Warning: best_losses do not match the last entry of best_loss_evolution\n",h5_file)
        best_fs = f['best_fs'][()]
        param_dict = {k: v[()] for k, v in f['param_dict'].items()}
    return tpl_vector, best_losses, param_dict, best_fs

def format_sci(v):
    if isinstance(v, (float, int)):
        exp = int(np.floor(np.log10(abs(v)))) if v != 0 else 0
        coeff = v / 10**exp if v != 0 else 0
    if abs(exp) > 1:
        return rf'${coeff:.1f}\times 10^{exp}$'
    else:
        return rf'${v:.0f}$'
    return str(v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate best losses across multiple directories to compute mean and 3-sigma credible intervals.")
    args = parser.parse_args()
    
    dirs = sorted(glob.glob("../apaper_results/*"))

    save_path = 'paper_results_tdi.h5'
    if os.path.exists(save_path):
        print(f"Loading aggregated results from {save_path}.")
        with h5py.File(save_path, 'r') as f:
            all_best_losses_noise = f['all_best_losses_noise'][()]
            tpl_vector = f['tpl_vector'][()]
            mean_losses = f['mean_losses'][()]
            lower_interval = f['lower_interval'][()]
            upper_interval = f['upper_interval'][()]
            best_fs = f['best_fs'][()]
    else:
        tpl_vector = None
        all_best_losses_noise = []
        best_fs_list = []
        
        for output_dir in dirs:
            h5_file = os.path.join(output_dir, 'best_results.h5')
            if not os.path.exists(h5_file):
                print(f"Warning: {h5_file} not found. Skipping.")
                continue

            current_tpl, current_losses, _, best_fs = load_best_results(h5_file)

            if tpl_vector is None:
                tpl_vector = current_tpl
            elif not np.array_equal(tpl_vector, current_tpl):
                raise ValueError(f"Tpl_vector mismatch in {output_dir}")

            if '1e-05' in output_dir:
                all_best_losses_noise.append(-current_losses)
                best_fs_list.append(best_fs)

        if not all_best_losses_noise:
            print("Error: No valid best_results.h5 files found.")
            exit(1)

        # Remove entries with mismatched length
        valid_losses = []
        for losses in all_best_losses_noise:
            if len(losses) == len(tpl_vector):
                valid_losses.append(losses)
            else:
                print(f"Warning: Skipping entry with mismatched length ({len(losses)} vs {len(tpl_vector)})")
        if not valid_losses:
            print("Error: No valid entries with matching tpl_vector length.")
            exit(1)
        
        all_best_losses_noise = np.array(valid_losses)
        best_fs = np.array(best_fs_list)

        mean_losses = np.median(all_best_losses_noise, axis=0)
        lower_interval = np.quantile(all_best_losses_noise, 0.0228, axis=0)  # ~2σ lower
        upper_interval = np.quantile(all_best_losses_noise, 0.9772, axis=0)  # ~2σ upper

        with h5py.File(save_path, 'w') as f:
            f.create_dataset('all_best_losses_noise', data=all_best_losses_noise)
            f.create_dataset('tpl_vector', data=tpl_vector)
            f.create_dataset('mean_losses', data=mean_losses)
            f.create_dataset('lower_interval', data=lower_interval)
            f.create_dataset('upper_interval', data=upper_interval)
            f.create_dataset('best_fs', data=best_fs)
        print(f"Aggregated results saved to {save_path}.")

    # Plot credible intervals
    plt.figure(figsize=(10, 6))
    plt.plot(tpl_vector, all_best_losses_noise.T, color='gray', alpha=0.3)
    plt.xlabel('Tpl')
    plt.ylabel('Detection Statistic')
    plt.legend()
    plt.grid(True)
    plt.savefig('detection_statistic_vs_tpl.pdf')

    # mean inital f
    mean_f = np.mean(best_fs[:,:,:5],axis=-1)
    mean_fdot = np.mean(np.gradient(best_fs,5e4,axis=-1)[:,:,:5],axis=-1)

    # Select representative Tpl values (e.g., last Tpl)
    num_tpl = len(tpl_vector)
    tpl_indices = [num_tpl//5, 2*num_tpl//5, 3*num_tpl//5, 3*num_tpl//5, 4*num_tpl//5]  # First, middle, last
    tpl_values = tpl_vector[tpl_indices]
    colors = ['C0', 'C1', 'C2']

    plt.figure(figsize=(10, 6))
    for idx, tpl_val in zip(tpl_indices, tpl_values):
        plt.plot(mean_f[:, idx], mean_fdot[:, idx], '.', label=f'Tpl = {tpl_val:.3f}', alpha=0.5)

    plt.xlabel('f [Hz]')
    plt.ylabel('fdot [Hz/s]')
    plt.legend()
    plt.grid(True)
    plt.savefig('frequency_pdf.pdf')

    plt.figure(figsize=(10, 6))
    for idx, tpl_val, cc in zip(tpl_indices, tpl_values, colors):
        neg_losses = (all_best_losses_noise[:, idx] - np.mean(all_best_losses_noise[:, idx]))/np.std(all_best_losses_noise[:, idx])

        hist, bin_edges, _ = plt.hist(neg_losses, bins='auto', label=f'Tpl = {tpl_val:.3f}', density=True, color=cc, alpha=0.5)
        
        # Plot PDF
        distro = gumbel_r(*gumbel_r.fit(neg_losses))
        plt.semilogy(bin_edges, distro.pdf(bin_edges), color=cc)

    plt.xlabel('Detection Statistic')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.savefig('noise_gumbel_fit.pdf')