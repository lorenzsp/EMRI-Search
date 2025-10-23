import h5py
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import glob
from scipy.stats import gumbel_r
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pickle
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, auc
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
from few.utils.constants import MTSUN_SI, YRSID_SI
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pastamarkers import pasta, salsa
from theoretical_pdet import detection_probability
from theoretical_pdet import detection_threshold as detection_threshold_func

from plot_styles import apply_physrev_style

# Apply the style
apply_physrev_style()

def chirpmass_from_f_fdot_few(f, fdot):
    """
    Calculate chirp mass using few constants.
    Returns chirp mass in solar masses.
    """
    M_chirp = ((10**f)**(-11) * (10**fdot)**3 * np.pi**(-8) * (5/96)**3)**(1/5) / MTSUN_SI
    return M_chirp

def get_detection_threshold(normalized, alpha, gumbel=True, list_hyp=False):
    """Compute detection threshold for given significance level alpha."""
    if gumbel:
        if list_hyp:
            detection_threshold = [gumbel_r(*gumbel_r.fit(el)).isf(alpha) for el in normalized.T]
        else:
            detection_threshold = gumbel_r(*gumbel_r.fit(np.max(normalized, axis=1))).isf(alpha)
    else:
        
        if list_hyp:
            detection_threshold = np.quantile(normalized, 1-alpha/len(tpl_vector), axis=0)
        else:
            detection_threshold = np.quantile(np.max(normalized, axis=1), 1-alpha)
    return detection_threshold

def compute_detection_probability(results, values, detection_threshold):
    """Compute detection probability for each unique value at given significance level."""
    unique_values = np.unique(values)
    unique_values = unique_values[unique_values > 1.0]  # only above 1
    detection_probs = []
    detection_std_probs = []

    print(f"Quantile for detection: {detection_threshold}")
    for val in unique_values:
        mask = np.isclose(values, val, rtol=1e-3)
        detections = []
        for r in np.array(results)[mask]:
            # Compute det_stat for each result
            det_stat = (r['losses'] - mean_noise)/std_noise
            # to use any
            # detected = np.any(det_stat > detection_threshold)
            det_stat = np.max(det_stat)  # use the max statistic across templates
            detected = det_stat > detection_threshold
            detections.append(detected)
        prob = np.mean(detections) if len(detections) > 0 else 0.0
        # Bernoulli standard deviation
        std_prob = np.sqrt(prob * (1 - prob) / len(detections)) if len(detections) > 0 else 0.0
        detection_probs.append(prob)
        detection_std_probs.append(std_prob)
        print(f"Detection probability for {val}: {prob} ± {std_prob}")
    return unique_values, np.asarray(detection_probs), np.asarray(detection_std_probs)

def compute_accuracy(results, values, detection_threshold):
    """Compute median relative frequency error for each unique value."""
    unique_values = np.unique(values)
    unique_values = unique_values[unique_values > 1.0]  # only above 1
    acc_medians = []
    acc_err_low = []
    acc_err_high = []

    for val in unique_values:
        mask = np.isclose(values, val, rtol=1e-3)
        accs = []
        for r in np.array(results)[mask]:
            # check if detected
            # detected = np.max((r['losses'] - mean_noise)/std_noise) > detection_threshold
            detected = np.any((r['losses'] - mean_noise)/std_noise > detection_threshold)
            if detected:
                # Find best tpl index based on max loss
                best_idx = np.argmax((r['losses'] - mean_noise)/std_noise)
                acc = r['rel_diff_medians'][best_idx]
                accs.append(acc)
        if len(accs) > 0:
            med = np.median(accs)
            low = np.percentile(accs, 16)
            high = np.percentile(accs, 84)
        else:
            med = low = high = np.nan
        acc_medians.append(med)
        acc_err_low.append(med - low)
        acc_err_high.append(high - med)
    return unique_values, np.array(acc_medians), np.array(acc_err_low), np.array(acc_err_high)

if __name__ == "__main__":
    from plot_best_results import load_best_results

    parser = argparse.ArgumentParser(description="Plot detection probability versus SNR and scatter plot for Tpl vs ef.")
    args = parser.parse_args()

    # Load noise distribution
    save_path = 'paper_results_tdi.h5'
    if not os.path.exists(save_path):
        print(f"Error: {save_path} not found.")
        exit(1)
    print(f"Loading aggregated results from {save_path}.")
    with h5py.File(save_path, 'r') as f:
        all_best_losses_noise = f['all_best_losses_noise'][()]
        tpl_vector = f['tpl_vector'][()]
        best_fs = f['best_fs'][()]
        noise_f = np.mean(best_fs[:,:,:5],axis=-1)
        noise_fdot = np.mean(np.gradient(best_fs,5e4,axis=-1)[:,:,:5],axis=-1)
    
    
    mean_noise = all_best_losses_noise.mean(axis=0)
    std_noise = all_best_losses_noise.std(axis=0)
    normalized = (all_best_losses_noise - mean_noise) / std_noise

    plt.figure()
    # Get colormap and normalize for number of templates
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(0, normalized.shape[1]-1)
    
    # Plot histograms and fitted Gumbel distributions for each template
    for ii in range(normalized.shape[1]):
        color = cmap(norm(ii))
        # Plot histogram
        # plt.hist(normalized[:,ii], bins=50, density=True, alpha=0.6, label=f'Template {ii+1}')
        
        # Fit Gumbel distribution and plot
        params = gumbel_r.fit(normalized[:,ii])
        x_range = np.linspace(normalized[:,ii].min(), normalized[:,ii].max(), 100)
        plt.plot(x_range, gumbel_r.pdf(x_range, *params), '-', linewidth=2, color=color, alpha=0.7)
    plt.semilogy()
    plt.xlabel('Normalized Statistic')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('gumbel_fits.pdf')
    
    # Try to load cached results if available, otherwise process and save

    cache_file = "paper_detection_cache.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, "rb") as f:
            results_detection, snr_values = pickle.load(f)
    else:
        # Process signal-injected realizations
        results_detection = []
        snr_values = []
        dirs = glob.glob("../apaper_results/*")
        
        if not dirs:
            print("Error: No signal-injected realization directories found.")
            exit(1)

        for output_dir in dirs:
            h5_file = os.path.join(output_dir, 'best_results.h5')
            if not os.path.exists(h5_file):
                print(f"Warning: {h5_file} not found. Skipping.")
                continue

            current_tpl, current_losses, param_dict, best_fs = load_best_results(h5_file)
            with h5py.File(h5_file, 'r') as f:
                snr = f['SNR'][()]
                rel_diff_medians = f['rel_diff_stats/medians'][()]  # Load medians from the group

            if not np.array_equal(tpl_vector, current_tpl):
                print(f"Warning: Tpl_vector mismatch in {output_dir}. Skipping.")
                continue

            f0 = best_fs[:,:5].mean(axis=-1)
            fdot0 = np.gradient(best_fs, 5e4, axis=-1)[:,:5].mean(axis=-1)
            param_dict["f0"] = f0
            param_dict["fdot0"] = fdot0
            param_dict["losses"] = -current_losses
            param_dict["snr"] = float(snr)
            param_dict["rel_diff_medians"] = rel_diff_medians  # Add rel_diff_medians to param_dict
            results_detection.append(param_dict)
            snr_values.append(float(snr))
            print(f"Processed {h5_file}: SNR={snr}")

        if not results_detection:
            print("Error: No valid signal-injected realizations processed.")
            exit(1)

        # Save to cache for future runs
        with open(cache_file, "wb") as f:
            pickle.dump((results_detection, snr_values), f)
        print(f"Saved processed results to {cache_file}")
    
    # Convert to arrays
    snr_values = np.array(snr_values)

    # Plot detection probability vs SNR for different significance levels
    # Create a figure with two subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [1.5, 1]})

    # Detection probability vs SNR for different significance levels
    alpha_values = [0.5, 0.01, 0.0001]  # Different significance levels
    colors = ['C0', 'C1', 'C2']  # Colors for different alpha values
    linestyles = ['-', '--', ':']  # Different line styles
    markers = ['o', 's', 'D']  # Different markers
    labels = [r'$p_{\rm FA}=0.5$', r'$p_{\rm FA}=10^{-2}$', r'$p_{\rm FA}=10^{-4}$']
    ms = 6  # Marker size

    mu0 = 632 * 2.
    sigma0 = np.sqrt(2 * 632 * 1.45**2)

    for alpha, color, ls, marker, lb in zip(alpha_values, colors, linestyles, markers, labels):
        detection_threshold = get_detection_threshold(normalized, alpha, gumbel=True)
        print(f"Detection threshold for alpha={alpha}: {detection_threshold}")
        unique_snrs, detection_probs_snr, detection_std_probs_snr = compute_detection_probability(results_detection, snr_values, detection_threshold)
        ax1.errorbar(unique_snrs, detection_probs_snr, yerr=detection_std_probs_snr, fmt=marker, color=color, linestyle=ls, alpha=0.7, capsize=3, ms=ms, label=lb, lw=2.)

        # Accuracy plot: Median relative frequency error vs SNR
        unique_snrs, acc_med, acc_err_l, acc_err_h = compute_accuracy(results_detection, snr_values, detection_threshold)
        ax2.errorbar(unique_snrs, acc_med, yerr=[acc_err_l, acc_err_h], fmt=marker, color=color, linestyle=ls, alpha=0.7, capsize=3, ms=ms, label=lb, lw=2.)
        print("Detection threshold:", detection_threshold, "Alpha:", alpha, "acc_med", acc_med)

    P_D_values = []
    A_test_values = np.linspace(20, 40, 100)
    snr_mismatch = (1.-0.0)**0.5  # assuming average mismatch of 0.5
    print("SNR mismatch factor:", snr_mismatch)
    N = 632
    pf_per_template = 1e-2 / 1e25
    PD_curve = np.array([detection_probability(A * snr_mismatch, 632, pf_per_template) for A in A_test_values])
    ax1.plot(A_test_values, PD_curve, '--', color='C5', lw=1.5, alpha=0.7, zorder=10)

    ax1.set_ylabel('Detection Probability')
    ax1.grid(True, axis='y')
    ax1.legend(title='False Alarm Probability', loc='lower right')
    ax1.set_ylim(-0.05, 1.05)

    idx_50 = np.argmin(np.abs(PD_curve - 0.02))
    x_annotate = A_test_values[idx_50]
    y_annotate = PD_curve[idx_50]

    ax1.annotate('Theoretical \nTemplate Bank\n $p_{\\mathrm{FA}}=10^{-2}$', 
                xy=(x_annotate, y_annotate), 
                xytext=(x_annotate + 2, y_annotate + 0.01),
                # arrowprops=dict(arrowstyle='->', color=colors[1], lw=1.5),
                fontsize=10, 
                color='C5')

    ax2.set_xlabel('SNR')
    ax2.set_ylabel('Relative Frequency Error')
    ax2.set_yscale('log')
    # ax2.legend()
    # ax2.set_ylim(3e-4, 1e-2)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('detection_and_accuracy_vs_snr.pdf')
    plt.close('all')
    
    ####################################################
    # Scatter plot Tpl vs ef, colored by SNR, markers by detection
    expected_snrs = [25, 30, 35]
    
    cmap = plt.get_cmap('inferno')
    cmap = plt.get_cmap(salsa.pesto)
    
    alpha = 0.001  # Use default alpha for scatter plot
    detection_threshold = get_detection_threshold(normalized, alpha)
    quantile_detection = np.quantile(normalized, 1-alpha/len(tpl_vector), axis=0)
    norm_ds = np.array([np.max((r['losses'] - mean_noise)/std_noise) for r in results_detection])
    detected = np.array([np.any(np.max((r['losses'] - mean_noise)/std_noise) > detection_threshold) for r in results_detection])

    m1_values = np.array([r['m1'] for r in results_detection])
    m2_values = np.array([r['m2'] for r in results_detection])
    tpl_values = np.array([r['Tpl'] for r in results_detection])
    ef_values = np.array([r['e0'] for r in results_detection])
    dist_values = np.array([r['dist'] for r in results_detection])
    f0_values = np.array([r['f0'][np.argmax(np.max((r['losses'] - mean_noise)/std_noise))] for r in results_detection])
    fdot0_values = np.array([r['fdot0'][np.argmax(np.max((r['losses'] - mean_noise)/std_noise))] for r in results_detection])
    
    M_chirp_values = chirpmass_from_f_fdot_few(f0_values, fdot0_values)
    M_chirp_noise = chirpmass_from_f_fdot_few(noise_f, noise_fdot)

    norm = plt.Normalize(1.0, 100.0)
    
    
    fig, ax = plt.subplots()
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$m_2$ [$M_\odot$]')
    for snr, mark in zip(expected_snrs, [pasta.penne, pasta.rigatoni, pasta.farfalle]):
        mask = np.isclose(snr_values, snr, rtol=1e-3)
        det_mask = mask & detected
        if np.any(det_mask):
            z_values = np.array([z_at_value(Planck18.luminosity_distance, d*u.Gpc) for d in dist_values[det_mask]])
            color_list = [cmap(norm(el)) for el in m2_values[det_mask]/(1+z_values)]
            plt.scatter(m1_values[det_mask]/(1+z_values), z_values, marker=mark, c=color_list, alpha=0.7)
            plt.semilogx()
            # plt.scatter(m1_values[det_mask]/(1+z_values), dist_values[det_mask], marker=mark, c=color_list, alpha=0.7)
    
    
    detected_marker = mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize=7, label='Detected')
    not_detected_marker = mlines.Line2D([], [], color='k', marker='x', linestyle='None', markersize=7, label='Not detected')

    # Create color patches for SNR legend
    snr_legend = [mlines.Line2D([], [], color='k', marker=mark, linestyle='None', markersize=7, label=f'SNR={snr}') for snr, mark in zip(expected_snrs, [pasta.penne, pasta.rigatoni, pasta.farfalle])]
    # Combine legends
    plt.legend(handles=snr_legend, loc='best')
    
    plt.xlabel(r'$m_1$ [$M_\odot$]')
    plt.ylabel(r'Redshift $z$')
    plt.tight_layout()
    plt.savefig('scatter.pdf')
    
    ###############################################
    # Build labels and scores for ROC

    labels = []
    scores = []
    argmax_scores = []
    print("Building labels and scores for ROC curve...")#, snr_values)
    dict_labels = {snr: [] for snr in snr_values}

    # Noise-only trials
    for noise_row in all_best_losses_noise:
        det_stat = (noise_row - mean_noise) / std_noise
        score = np.max(det_stat)  # use the max statistic across time
        argmax_scores.append(np.argmax(det_stat))
        labels.append(0)
        scores.append(score)
    
    # show argmax histogram
    plt.figure()
    plt.hist(argmax_scores, bins=len(tpl_vector), density=True, alpha=0.7)
    plt.xlabel('Template Index of Max Score (Noise)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('argmax_histogram_noise.pdf')
    

    # Signal+noise trials
    for r in results_detection:
        det_stat = (r['losses'] - mean_noise) / std_noise
        score = np.max(det_stat)
        # define label as the SNR value
        labels.append(r['snr'])
        scores.append(score)

    labels = np.array(labels)
    scores = np.array(scores)

    # histogram of scores
    plt.figure()
    # Define log-spaced bins for better visualization
    min_score = np.min(np.array(scores)[np.array(labels)==0])
    max_score = np.max(np.array(scores)[np.array(labels)==0])
    log_bins = np.logspace(np.log10(min_score), np.log10(max_score), 10)
    plt.hist(np.array(scores)[np.array(labels)==0], bins=log_bins, label='Noise', alpha=0.7, density=True)
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    for i, snr in enumerate([20, 30, 40]):
        min_score = np.min(np.array(scores)[np.array(labels)==snr])
        max_score = np.max(np.array(scores)[np.array(labels)==snr])
        log_bins = np.logspace(np.log10(min_score), np.log10(max_score), 10)
        plt.hist(np.array(scores)[np.array(labels)==snr], bins=log_bins, label=f'SNR$=${snr}', alpha=1.0, density=True, histtype='step', linewidth=2.5, linestyle=linestyles[i])
    plt.semilogx()
    plt.semilogy()
    plt.axvline(get_detection_threshold(normalized, 0.5), color='grey', linestyle='-', label=r'$p_{\rm FA}=0.5$')
    plt.axvline(get_detection_threshold(normalized, 0.01), color='grey', linestyle='--', label=r'$p_{\rm FA}=10^{-2}$')
    plt.axvline(get_detection_threshold(normalized, 0.0001), color='grey', linestyle=':', label=r'$p_{\rm FA}=10^{-4}$')
    
    # Add text annotations next to the vertical lines
    from max_of_distribution import compute_max_stats
    MU_K = 2.0
    SIGMA_K = 1.45
    N = int(YRSID_SI / 5e4)
    Nopt = 500 * 512
    mu0 = N * MU_K
    sigma0 = np.sqrt(2 * N * SIGMA_K**2)
    results_per_seg = compute_max_stats(mu0, sigma0, Nopt, method='asymptotic')
    print("noise approx", results_per_seg['mean'], results_per_seg['variance']**0.5, "approx", mean_noise[-1], std_noise[-1])
    print("Relative difference", (results_per_seg['mean'] - 2 * mean_noise[-1]) / (2 * mean_noise[-1]))

    for A in [1e-5, 20, 30, 40]:
        snr = A
        mu1 = N * MU_K + A**2
        sigma1 = np.sqrt(2 * N * SIGMA_K**2 + 4 * A**2)
        results_per_seg = compute_max_stats(mu1, sigma1, Nopt, method='asymptotic')
        max_last_seg = np.asarray([r['losses'][-1] for r in results_detection if r['snr'] == snr])
        print(f"SNR={snr}")
        print("Signal+noise approx", results_per_seg['mean'], results_per_seg['variance']**0.5, 2 * np.mean(max_last_seg), np.std(max_last_seg))
        print("Relative difference", (results_per_seg['mean'] - 2 * np.mean(max_last_seg)) / (2 * np.mean(max_last_seg)), (results_per_seg['variance']**0.5 - np.std(max_last_seg)) / np.std(max_last_seg))

    # plot fit
    log_bins = np.logspace(-1, 3, 50)
    print(gumbel_r.fit(np.max(normalized, axis=1)))
    gumb = gumbel_r(*gumbel_r.fit(np.max(normalized, axis=1))).pdf(log_bins)
    plt.plot(log_bins, gumb, '-', linewidth=2, color='C0', alpha=0.7)
    
    plt.xlabel(r'Normalized Statistic $\mathcal{S}$ ')
    plt.ylabel('Density')
    plt.legend(ncol=1)
    # plt.title('Histogram of Detection Statistic Scores')
    plt.ylim(1e-4, 10)
    plt.xlim(0.6, 2000)
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig('score_histogram.pdf')

    # Plot ROC
    # Full ROC with inset
    fig, ax = plt.subplots()

    # Inset axes for zoomed region
    axins = inset_axes(ax, width="50%", height="50%", loc='lower left',
                    bbox_to_anchor=(0.45,0.08,0.5,0.5), bbox_transform=ax.transAxes)

    # Main ROC
    for snr in [25, 30, 35]:
        # select scores
        new_scores = scores[(labels == snr) | (labels == 0.0)]
        new_labels = labels[(labels == snr) | (labels == 0.0)]
        new_labels = np.array([1 if l > 0 else 0 for l in new_labels])  # binary labels: 1 for signal, 0 for noise
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(new_labels, new_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'SNR={snr}')# (AUC = {roc_auc:.3f})
        
        axins.plot(fpr, tpr, lw=2)
        axins.set_xlim([5e-4, 5e-2])   # zoomed FPR range
        axins.set_ylim([0.5, 1.02])
        axins.set_xscale('log')
        # Increase number of y-ticks
        axins.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        # axins.set_title("Low-FPR zoom", fontsize=9)
        axins.grid(True, which="both")

    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    ax.plot([0,1],[0,1], lw=1, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title('ROC curve for GW search pipeline')
    ax.legend(loc='lower right')
    plt.tight_layout()
    ax.grid(True)

    plt.savefig('roc_curve.pdf')
    
    