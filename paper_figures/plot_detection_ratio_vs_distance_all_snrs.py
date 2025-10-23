import h5py
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from plot_best_results import load_best_results
from plot_detection_probability import get_detection_threshold

def main():
    # List all SNRs available in the results directories
    dirs = glob.glob("../apaper_results/*snr*")
    snr_set = set()
    for d in dirs:
        base = os.path.basename(d)
        try:
            snr_val = float(base.split('snr')[-1])
            if snr_val > 1.0:
                snr_set.add(snr_val)
        except Exception:
            continue
    snr_list = sorted(list(snr_set))
    print(f"Found SNRs: {snr_list}")

    # Prepare ratio data for each SNR
    ratio_dict = {}
    param = 'dist'  # You can change to any parameter of interest
    bins = 6
    for snr in snr_list:
        cache_file = f"paper_scatter_cache_snr{snr}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                results_detection, _ = pickle.load(f)
        else:
            results_detection = []
            snr_values = []
            dirs_snr = glob.glob(f"../apaper_results/*snr{snr}*")
            for output_dir in dirs_snr:
                h5_file = os.path.join(output_dir, 'best_results.h5')
                if not os.path.exists(h5_file):
                    continue
                current_tpl, current_losses, param_dict, best_fs = load_best_results(h5_file)
                try:
                    with h5py.File(h5_file, 'r') as f:
                        snr_val = f['SNR'][()]
                except Exception:
                    continue
                param_dict["losses"] = -current_losses
                param_dict["snr"] = float(snr_val)
                results_detection.append(param_dict)
                snr_values.append(float(snr_val))
            with open(cache_file, "wb") as f:
                pickle.dump((results_detection, snr_values), f)
        if not results_detection:
            continue
        # Compute detection threshold (use noise file from first SNR)
        if snr == snr_list[0]:
            noise_file = "paper_results_tdi.h5"
            with h5py.File(noise_file, 'r') as f:
                all_best_losses_noise = f['all_best_losses_noise'][()]
            mean_noise = all_best_losses_noise.mean(axis=0)
            std_noise = all_best_losses_noise.std(axis=0)
            normalized = (all_best_losses_noise - mean_noise) / std_noise
            detection_threshold = get_detection_threshold(normalized, 1e-2)
        detected = np.array([np.max((r['losses'] - mean_noise)/std_noise) > detection_threshold for r in results_detection])
        not_detected = ~detected
        # Get parameter values
        param_values = np.array([r[param] for r in results_detection])
        # Bin edges (log for dist)
        min_val = np.min(param_values)
        max_val = np.max(param_values)
        log_bins = np.logspace(np.log10(min_val), np.log10(max_val), bins+1)
        detected_data = param_values[detected]
        not_detected_data = param_values[not_detected]
        hist_detected, bin_edges = np.histogram(detected_data, bins=log_bins)
        hist_notdet, _ = np.histogram(not_detected_data, bins=log_bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = hist_detected / np.where(hist_notdet == 0, np.nan, hist_notdet)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ratio_dict[snr] = (bin_centers, ratio)

    # Plot ratio for all SNRs
    plt.figure(figsize=(8, 5))
    for snr, (bin_centers, ratio) in ratio_dict.items():
        plt.plot(bin_centers, ratio, marker='o', label=f'SNR={snr}')
    plt.xscale('log')
    plt.xlabel('Luminosity Distance [Gpc]')
    plt.ylabel('Detected / Not Detected Ratio')
    plt.title('Detection Ratio vs Distance for Different SNRs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('detection_ratio_vs_distance_all_snrs.pdf')
    plt.show()

if __name__ == "__main__":
    main()
