import h5py
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import glob
from plot_best_results import load_best_results
from plot_detection_probability import get_detection_threshold
import pickle
import corner
from matplotlib.patches import Rectangle
from plot_styles import apply_physrev_style
from astropy.cosmology import Planck18
from astropy.cosmology import z_at_value
from astropy import units as u
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
from matplotlib.lines import Line2D
import pandas as pd
# Apply the style
apply_physrev_style()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scatter plot Tpl vs ef for a given SNR.")
    parser.add_argument("--snr", type=int, default=30, help="SNR value to plot")
    parser.add_argument("--alpha", type=float, default=1e-4, help="Significance level for detection threshold")
    parser.add_argument("--noise_file", type=str, default="paper_results_tdi.h5", help="Noise file path")
    args = parser.parse_args()

    # Load noise distribution
    if not os.path.exists(args.noise_file):
        print(f"Error: {args.noise_file} not found.")
        exit(1)
    with h5py.File(args.noise_file, 'r') as f:
        all_best_losses_noise = f['all_best_losses_noise'][()]
        tpl_vector = f['tpl_vector'][()]
    mean_noise = all_best_losses_noise.mean(axis=0)
    std_noise = all_best_losses_noise.std(axis=0)
    normalized = (all_best_losses_noise - mean_noise) / std_noise

    # Load cached results
    results_detection = []
    snr_values = []
    dirs = glob.glob(f"../apaper_results/*snr{args.snr}*")
    
    cache_file = f"paper_scatter_cache_{args.snr}.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")

        with open(cache_file, "rb") as f:
            results_detection, snr_values = pickle.load(f)
    else:
        print(f"No cache found. Processing directories matching 'random_search_*snr{args.snr}*'")
        for output_dir in dirs:
            h5_file = os.path.join(output_dir, 'best_results.h5')
            if not os.path.exists(h5_file):
                print(f"Warning: {h5_file} not found. Skipping.")
                continue

            current_tpl, current_losses, param_dict, best_fs = load_best_results(h5_file)
            try:
                with h5py.File(h5_file, 'r') as f:
                    snr = f['SNR'][()]
                    rel_diff_medians = f['rel_diff_stats/medians'][()]  # Load medians from the group
            except Exception as e:
                print(f"Warning: Failed to read SNR or rel_diff_medians from {h5_file}. Skipping. Error: {e}")
                continue

            if not np.array_equal(tpl_vector, current_tpl):
                print(f"Warning: Tpl_vector mismatch in {output_dir}. Skipping.")
                continue

            param_dict["losses"] = -current_losses
            param_dict["snr"] = float(snr)
            param_dict["rel_diff_medians"] = rel_diff_medians  # Add rel_diff_medians to param_dict
            f0 = best_fs[:,:5].mean(axis=-1)
            fdot0 = np.gradient(best_fs, 5e4, axis=-1)[:,:5].mean(axis=-1)
            param_dict["f0"] = f0
            param_dict["fdot0"] = fdot0
            param_dict["normalized_ds"] = np.max((param_dict['losses'] - mean_noise)/std_noise)
            # param_dict["normalized_ds"] = best_fs[:,(param_dict['losses'] - mean_noise)/std_noise).argmax()]
            results_detection.append(param_dict)
            snr_values.append(float(snr))
            print(f"Processed {h5_file}: SNR={snr}")

        with open(cache_file, "wb") as f:
            pickle.dump((results_detection, snr_values), f)
        print(f"Saved processed results to {cache_file}")
    

    results_detection = []
    snr_values = []
    for snr in [20, 25, 30, 35, 40]:
        with open(f"paper_scatter_cache_{snr}.pkl", "rb") as f:
            results_, snr_ = pickle.load(f)
            print(f"Loaded {len(results_)} results for SNR={snr}")
            results_detection.extend(results_)
            snr_values.extend(snr_)

    snr_values = np.array(snr_values)
    # Compute detection threshold
    detection_threshold = get_detection_threshold(normalized, args.alpha)
    print(f"Computed detection threshold: {detection_threshold:.2f}")
    detected = np.array([np.max((r['losses'] - mean_noise)/std_noise) > detection_threshold for r in results_detection])
    print(f"Detection threshold at alpha={args.alpha}: {detection_threshold:.2f}")
    print(f"Number of detected signals: {np.sum(detected)} out of {len(detected)}")
    norm_ds = np.asarray([np.max((r['losses'] - mean_noise)/std_noise) for r in results_detection])

    m1_values = np.array([r['m1'] for r in results_detection])
    m2_values = np.array([r['m2'] for r in results_detection])
    distances = np.array([r['dist'] for r in results_detection])
    p0_values = np.array([r['p0'] for r in results_detection])
    e0_values = np.array([r['e0'] for r in results_detection])
    ef_values = np.array([r['ef'] for r in results_detection])
    phiS_values = np.array([r['phiS'] for r in results_detection])
    thetaS_values = np.array([r['qS'] for r in results_detection])
    f0_values = np.array([r['f0'][np.argmax(np.max((r['losses'] - mean_noise)/std_noise))] for r in results_detection])
    fdot0_values = np.array([r['fdot0'][np.argmax(np.max((r['losses'] - mean_noise)/std_noise))] for r in results_detection])
    # redefined f0 and fdot0  in results_detection
    semimaj_axis = p0_values / (1 - e0_values**2)  # Semi-major axis from p0 and e0
    for res, f0, fdot0 in zip(results_detection, f0_values, fdot0_values):
        res['f0'] = f0
        res['fdot0'] = fdot0
        semi_meters = res['p0'] / (1 - res['e0']**2) * res['m1'] * 1476.6250385167637  # in meters
        res['semimaj'] = semi_meters / 3.085677581491367e+16  # in parsec
    
    # Scatter plot for the requested SNR
    mask = np.isclose(snr_values, args.snr, rtol=1e-3)
    mask = snr_values > 1
    det_mask = mask & detected
    not_det_mask = mask & ~detected
    
    # Get filtered data
    filtered_distances = distances[det_mask]
    z_values = np.array([z_at_value(Planck18.luminosity_distance, d*u.Gpc) for d in filtered_distances])
    filtered_z = z_values #*1e3  # Convert Gpc to Mpc
    filtered_m1 = m1_values[det_mask]/(1 + z_values)  # Redshifted mass
    filtered_m2 = m2_values[det_mask]/(1 + z_values)  # Redshifted mass
    
    z_values_notdet = np.array([z_at_value(Planck18.luminosity_distance, d*u.Gpc) for d in distances[not_det_mask]])
    filtered_m1_notdet = m1_values[not_det_mask]/(1 + z_values_notdet)  # Redshifted mass
    filtered_m2_notdet = m2_values[not_det_mask]/(1 + z_values_notdet)  # Redshifted mass

    filtered_detected = detected[det_mask]
    filtered_thetaS = thetaS_values[det_mask]
    filtered_phiS = phiS_values[det_mask]
    filtered_semimaj = semimaj_axis[det_mask]
    filtered_ef = ef_values[det_mask]  # Final eccentricity for coloring
    filtered_e0 = np.array([r['e0'] for r in np.array(results_detection)[det_mask]])  # Initial eccentricity

    # Create 2D plot with luminosity distance on x-axis and broken y-axis
    import matplotlib.patches as patches
    
    # Create figure with two subplots sharing y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*6.5, 3.7), sharey=True)
    
    # Define the break range - adjust these values based on your data
    x_break_low = 100.0  # End of left subplot
    x_break_high = 1.0e5  # Start of right subplot
    
    # Scaling factors for marker sizes based on masses
    m1_scale = 500
    m2_scale = 50
    
    m1_sizes = m1_scale * (filtered_m1 / filtered_m1.max())
    m2_sizes = m2_scale * (filtered_m2 / filtered_m2.max())
    
    # Separate detected and not detected for cleaner plotting
    detected_indices = np.where(filtered_detected)[0]
    not_detected_indices = np.where(~filtered_detected)[0]

    # Plot on both axes
    # for ax in [ax1, ax2]:
        # Primary black holes (detected) - colored by final eccentricity
    scatter1 = ax2.scatter(filtered_m1[detected_indices], filtered_z[detected_indices],
                c=filtered_e0[detected_indices], 
                alpha=0.7,
                marker='o', 
                cmap='plasma', vmin=filtered_e0.min(), vmax=filtered_e0.max(),
                label='EMRI Detections'
                )
        
        # Secondary black holes (detected) - colored by initial eccentricity
    scatter2 = ax1.scatter(filtered_m2[detected_indices], filtered_z[detected_indices],
                c=filtered_ef[detected_indices], 
                alpha=0.7,
                marker='s', 
                cmap='cividis', vmin=filtered_ef.min(), vmax=filtered_ef.max(),
                label='EMRI Detections'
                )
        
    # ax1.scatter(filtered_m2_notdet, z_values_notdet, 
    #             map='cividis', vmin=filtered_ef.min(), vmax=filtered_ef.max(),
    #             alpha=0.7,marker='X', )
    # ax2.scatter(filtered_m1_notdet, z_values_notdet, 
    #             cmap='plasma', vmin=filtered_e0.min(), vmax=filtered_e0.max(),
    #             alpha=0.7,marker='X', )

    
    gw_data_path = '../gw_catalog_data/gw_catalog_masses.csv'
    lvk_data = pd.read_csv(gw_data_path)

    redshift_lvk = np.asarray([z_at_value(Planck18.luminosity_distance, d*u.Mpc) for d in lvk_data['luminosity_distance']])
    m1_lvk = lvk_data['primary_mass']
    m2_lvk = lvk_data['secondary_mass']
    ax1.scatter(m2_lvk, redshift_lvk, marker='.', s=5, c='red', label='LVK Detections')
    ax1.scatter(m1_lvk, redshift_lvk, marker='.', s=5, c='red')
    ax1.legend(loc='upper right', framealpha=1.0)
    
    
    # https://arxiv.org/pdf/2404.00941
    masses_qpe = np.asarray([1.2, 0.55, 0.55, 3.1, 42.5, 1.8, 5.5, 0.595, 6.55, 88.0, 5.8])
    z_qpe     = np.asarray([0.0181, 0.0505, 0.0175, 0.024, 0.044, 0.0237, 0.042, 0.13, 0.0206, 0.0136, 0.0053])
    ax2.scatter(masses_qpe * 1e6, z_qpe, marker='+', s=100, c='C3', label='QPEs \& QPOs')
    ax2.legend(loc='upper right', framealpha=1.0)
    # # Add colorbars
    # cbar1 = plt.colorbar(scatter2, ax=ax1, shrink=0.6, pad=-0.3, orientation='horizontal')
    # ax1.text(60.0, 2.5, 'Final Eccentricity ($e_f$)')
    cbar1_ax = fig.add_axes([0.1, 0.95, 0.35, 0.03])  # [left, bottom, width, height] in figure coordinates
    cbar1 = plt.colorbar(scatter2, cax=cbar1_ax, orientation='horizontal')
    cbar1_ax.set_title('Final Eccentricity ($e_f$)')
    cbar1_ax.xaxis.set_ticks_position('top')
    cbar1_ax.xaxis.set_label_position('top')

    # cbar2 = plt.colorbar(scatter1, ax=ax2, shrink=0.6, pad=-0.15, orientation='horizontal')
    # cbar2.set_label('Initial Eccentricity ($e_0$)')
    # ax2.text(1e6, 2.5, 'Initial Eccentricity ($e_0$)')
    cbar2_ax = fig.add_axes([0.55, 0.95, 0.35, 0.03])  # [left, bottom, width, height] in figure coordinates
    cbar2 = plt.colorbar(scatter1, cax=cbar2_ax, orientation='horizontal')
    cbar2_ax.set_title('Initial Eccentricity ($e_0$)')
    cbar2_ax.xaxis.set_ticks_position('top')
    cbar2_ax.xaxis.set_label_position('top')


    # Set x-limits for each subplot
    ax1.set_xlim(1, x_break_low)  # Lower range
    ax2.set_xlim(x_break_high, filtered_m1.max() * 1.7)  # Upper range
    
    # Set log scale for both x-axes
    # ax1.set_xscale('log')
    ax2.set_xscale('log')
    
    axtwin = ax2.twinx()  # Create a twin Axes sharing the y-axis
    axtwin.plot(filtered_z, filtered_distances, alpha=0)  # Invisible plot to set the scale
    axtwin.spines['left'].set_visible(False)
    axtwin.tick_params(labelleft=False, left=False)  # hide tick labels and ticks on the left
    # Hide the spines between ax1 and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # ax1.tick_params(labelright=False, right=False)  # hide tick labels and ticks on the right
    ax2.tick_params(labelleft=False, left=False)  # hide tick labels and ticks on the left

    # Add diagonal lines to indicate the break
    d = .015  # size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)        # bottom-right diagonal
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)      # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the right axes
    ax2.plot((-d, +d), (-d, +d), **kwargs)          # bottom-left diagonal
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)        # top-left diagonal

    # Labels and formatting
    ax1.set_xlabel('Source frame secondary mass [M$_\odot$]')
    ax2.set_xlabel('Source frame primary mass [M$_\odot$]')

    axtwin.set_ylabel('Luminosity Distance [Gpc]')
    ax1.set_ylabel('Redshift')
    
    # Add legend to left subplot
    # ax1.legend(loc='upper right')
    ax2.set_xlim(ax2.get_xlim()[0], ax2.get_xlim()[1]*1.5)
    
    # Add grid to both subplots
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'2d_mass_distance_broken_axis_snr_{args.snr}.pdf', bbox_inches='tight', dpi=300)
    
    ########################################################
    # # Scatter plot distance vs norm_ds
    # plt.figure(figsize=(6.5, 4))
    # if np.any(det_mask):
    #     plt.scatter(m1_values[det_mask], m2_values[det_mask], color='C0', label='Detected', alpha=0.7)
    # if np.any(not_det_mask):
    #     plt.scatter(m1_values[not_det_mask], m2_values[not_det_mask], color='C1', label='Not Detected', alpha=0.7)
    # plt.xscale('log')
    # plt.title(f'Scatter Plot of Distance vs Normalized Detection Statistic (SNR={args.snr})')
    # plt.legend()    
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f'scatter_distance_snr_{args.snr}.pdf')
    ########################################################
    # select only results with the given snr
    results_detection = [r for r in results_detection if r['snr'] == args.snr]
    det_mask = np.array([np.max((r['losses'] - mean_noise)/std_noise) > detection_threshold for r in results_detection])
    not_det_mask = ~det_mask

    # Create corner plot
    # Prepare data for corner plot
    params = ['m1', 'm2', 'a', 'ef', 'Tpl', 'dist','f0', 'fdot0', 'p0', 'e0']  # Add more parameters as needed
    param_labels = [r'$\log_{10} m_1/M_\odot$', r'$m_2 [M_\odot]$', r'$a$', r'$e_f$', r"$T_{\rm pl}$ [yr]", r'$d_{L}$ [Gpc]', r'$f_0$ [mHz]', r'$\dot{f}_0 \times 10^{10}$ [Hz$^2$]', r'$p_0$ [M]', r'$e_0$']
    # params = ['a', 'ef']
    # param_labels = [r'$a$', r'$e_f$']
    # Get data for detected and non-detected
    if np.any(det_mask):
        detected_data = np.stack([np.array([r[pp] for r in np.array(results_detection)[det_mask]]) for pp in params]).reshape(len(params),-1)
        detected_data = detected_data.T
    
    if np.any(not_det_mask):
        not_detected_data = np.stack([np.array([r[pp] for r in np.array(results_detection)[not_det_mask]]) for pp in params]).reshape(len(params),-1)
        not_detected_data = not_detected_data.T
    
    detected_data[:, 0] = np.log10(detected_data[:, 0])  # Log scale for m1
    not_detected_data[:, 0] = np.log10(not_detected_data[:, 0])  # Log scale for m1
    # apply mHz and 1e-10 Hz/s scaling
    not_detected_data[:, 6] = not_detected_data[:, 6]*1e3  # f0 in mHz
    detected_data[:, 6] = detected_data[:, 6]*1e3  # f0 in mHz
    not_detected_data[:, 7] = not_detected_data[:, 7]*1e10  # fdot in 1e-10 Hz/s
    detected_data[:, 7] = detected_data[:, 7]*1e10  # fdot in 1e-10 Hz/s
    # 1-ef
    # detected_data[:, 0] =  
    # not_detected_data[:, 2] = 1 - not_detected_data[:, 2]
    # Create figure with subplots
    fig = plt.figure(figsize=(6.5, 6.5))
    plt.plot(detected_data[:,0], detected_data[:,1], 'o', color='C0', alpha=0.7, label='Detected')
    plt.plot(not_detected_data[:,0], not_detected_data[:,1], 'o', color='C1', alpha=0.7, label='Not Detected')
    plt.xlabel(param_labels[0])
    plt.ylabel(param_labels[1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'scatter_a_ef_snr_{args.snr}.pdf')
    
    fig = plt.figure()
    if np.any(det_mask):
        corner.corner(detected_data, labels=param_labels, color='C0', fig=fig, bins=30,
                     hist_kwargs={'alpha': 0.7}, scatter_kwargs={'alpha': 0.9, 's': 15, 'edgecolors': 'black', 'linewidths': 0.3},
                     plot_density=False, plot_contours=False, fill_contours=False,
                     plot_datapoints=True, no_fill_contours=True)
    
    # if np.any(not_det_mask):
    #     corner.corner(not_detected_data, labels=param_labels, color='C1', fig=fig, bins=30,
    #                  hist_kwargs={'alpha': 0.7}, scatter_kwargs={'alpha': 0.9, 's': 15, 'edgecolors': 'black', 'linewidths': 0.3},
    #                  plot_density=False, plot_contours=False, fill_contours=False,
    #                  plot_datapoints=True, no_fill_contours=True)
    
    plt.tight_layout()
    plt.savefig(f'corner_snr_{args.snr}.pdf')
    # 
    ########################################################
    # Set exact number of ticks

    # Create horizontal subplots for parameter histograms
    fig, axes = plt.subplots(2, 5, figsize=(13, 6), sharey=True)
    bins = 3
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i % 2, i // 2]
        # Create log-spaced bins in the original space, then convert to log space
        min_val = np.min([detected_data[:, i].min(), not_detected_data[:, i].min()])
        max_val = np.max([detected_data[:, i].max(), not_detected_data[:, i].max()])

        if np.any(det_mask):
            data_to_plot = detected_data[:, i]
            
            # Use log-spaced bins for distance parameter
            if param in ['dist']:#, 'e0']:#, 'fdot0', 'f0']:        
                log_bins = np.logspace(np.log10(min_val), np.log10(max_val), bins+1)
                hist_detected, bin_edges = np.histogram(data_to_plot, bins=log_bins)
                ax.hist(data_to_plot, bins=log_bins, alpha=0.7, color='C0', 
                    label='Detected', density=False)
                ax.set_xscale('log')
            else:
                lin_bins = np.linspace(min_val, max_val, bins+1)
                hist_detected, bin_edges = np.histogram(data_to_plot, bins=lin_bins)
                ax.hist(data_to_plot, bins=lin_bins, alpha=0.7, color='C0',
                    label='Detected', density=False)
        
        if np.any(not_det_mask):
            data_to_plot = not_detected_data[:, i]
            
            if param in ['dist']:#, 'e0']:#, 'fdot0', 'f0']:
                log_bins = np.logspace(np.log10(min_val), np.log10(max_val), bins+1)
                hist_notdet, _ = np.histogram(data_to_plot, bins=log_bins)
                ax.hist(data_to_plot, bins=log_bins, alpha=0.7, color='C1',
                    label='Not Detected', density=False, histtype='step', linewidth=3.0)
                ax.set_xscale('log')
                ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
            else:
                lin_bins = np.linspace(min_val, max_val, bins+1)
                hist_notdet, _ = np.histogram(data_to_plot, bins=lin_bins)
                ax.hist(data_to_plot, bins=lin_bins, alpha=0.7, color='C1',
                    label='Not Detected', density=False, histtype='step', linewidth=3.0)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # ~5 ticks

        # Plot ratio of detected / not detected as a function of bin centers
        if np.any(det_mask) and np.any(not_det_mask):
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = hist_detected / np.where(hist_notdet == 0, np.nan, hist_notdet)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax2 = ax.twinx()
                ax2.plot(bin_centers, ratio, 'ko', label='Detected/Not Detected Ratio')
                
                # Set y-label only for the last column in each row
                if i // 2 == axes.shape[1] - 1:
                    ax2.set_ylabel('Detection Ratio')
                    ax2.tick_params(axis='y')
                else:
                    ax2.set_yticklabels([])
                
                ax2.set_ylim(bottom=0)
                ax2.set_ylim(0, 6.0)
                ax2.grid(False)
                ax2.hlines(1.0, bin_edges[0], bin_edges[-1], colors='gray', linestyles='dashed', alpha=0.9)
                
        
        xlabel = label
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=70.0)
    axes[0,0].set_ylabel('Event Count')
    axes[1,0].set_ylabel('Event Count')
    # Add legend for event types
    axes[0, 0].legend()
    # Add a dummy line for the detection ratio to the legend
    # Add a dot marker for the detection ratio to the legend
    
    axes[0, 0].legend(handles=axes[0, 0].get_legend_handles_labels()[0])
    detection_ratio = Line2D([0], [0], color='k', marker='o', linestyle='None', label='Ratio')
    detection_ratio_line = Line2D([0], [0], color='gray', linestyle='dashed', label='Ratio=1')
    
    axes[0, 4].legend(handles=[detection_ratio_line]+[detection_ratio])
    plt.tight_layout()
    plt.savefig(f'histograms_snr_{args.snr}.pdf')
    