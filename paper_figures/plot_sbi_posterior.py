"""
Plot SBI posterior samples with custom styling using pastamarkers.

This script loads posterior samples from posterior_samples_round_final.npy
and creates a corner plot with custom colors.
export PATH="/Library/TeX/texbin:$PATH"
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

try:
    from pastamarkers import pasta, salsa
    HAS_PASTA = True
except ImportError:
    print("Warning: pastamarkers not found. Using default colors.")
    HAS_PASTA = False

from plot_styles import apply_physrev_style

# Apply the style
apply_physrev_style()

# # Load posterior samples
# print("Loading posterior samples from posterior_samples_round_final.npy")
# samples_post = np.load('posterior_samples_round_final.npy')
# print(f"Loaded {len(samples_post)} samples with {samples_post.shape[1]} parameters")

root = "./"
# load follow-up evaluation results
data = np.load(root + "/sample_lls_n1000000_b100_p8.npz")
sample_lls, samples = data["sample_lls"], data["samples"]

# mcmc
mcmc_results = h5py.File(root + "/mcmc_emri_followup.h5", 'r')
loglike = mcmc_results['mcmc']['log_like'][:,0,].flatten()
samp_mcmc = mcmc_results['mcmc']['chain']['emri'][:,0].reshape(-1,5)

print("Shapes:", samples.shape, sample_lls.shape, samp_mcmc.shape)
# Load data from the specified HDF5 file
with h5py.File(root + '/best_results.h5', 'r') as f:
    params = {key: f[key][()] for key in f['param_dict'].keys()}

# true
m1_true = params['m1']
log10_m1_true = np.log10(m1_true)  # Convert to log10(m1)
m2_true = params['m2']
a_true = params['a']
Tpl_true = params['Tpl']
ef_true = params['ef']
truths = np.asarray([log10_m1_true, m2_true, a_true, Tpl_true, ef_true])
# truths = np.append(truths, np.nan)

# Define parameter labels
labels = [
    r'$\log_{10}(m_1/M_\odot)$', 
    r'$m_2$ [$M_\odot$]', 
    r'$a$', 
    r'$T_{\rm pl}$ [yr]', 
    r'$e_f$'
]

# Create corner plot
# Increase label size
plt.rcParams.update({
    'font.size': 30,
    'axes.labelsize': 30,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26
})

print("Creating corner plot...")

# Define xlim for each parameter (adjust as needed)
xlims = [
    (np.log10(5e5), 6.3),    # log10(m1)
    (20, 50),      # m2
    (0.0, 1.0),    # a
    (0.85, 0.93),    # Tpl
    (0.0, 0.1),   # ef
]

fig = corner.corner(
    samples,
    weights=np.ones_like(samples[:, 0])/len(samples[:, 0]),
    labels=labels,
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    # show_titles=True,
    plot_contours=True,
    max_n_ticks=4,
    color='C0',
    labelpad=0.2,
    hist_kwargs=dict(linewidth=1.5, histtype='step'),
)

threesigma = (1 - 0.9973)/2
fig = corner.corner(
    samp_mcmc,
    bins=10,
    weights=np.ones_like(samp_mcmc[:, 0])/len(samp_mcmc[:, 0]),
    levels=1 - np.exp(-0.5 * np.array([3]) ** 2),
    # labels=[r'$\log_{10} m_1$ [$M_\odot$]', r'$m_2$ [$M_\odot$]', r'$a$',  r'$T_{\rm pl}$',  r'$e_f$'],
    fill_contours=True,
    plot_datapoints=False,
    title_quantiles=(threesigma , 0.5, 1-threesigma),
    show_titles=True,
    title_fmt=".4f",
    title_kwargs={"fontsize": 22, "pad": 12, "color": 'C5'},
    hist_kwargs=dict(histtype='stepfilled'),
    color='C5',
    # truths=truths,
    max_n_ticks=4,
    labelpad=0.2,
    fig=fig
)
# Manually add truth values as vertical lines
print("Adding truth values to plot...")
axes = fig.get_axes()
n_params = len(truths)

# Add vertical lines for diagonal plots (marginal distributions)
for i in range(n_params):
    ax = axes[i * n_params + i]  # Diagonal element
    ax.axvline(truths[i], color='k', linestyle='--', linewidth=1, alpha=0.8)
    # Set xlim for diagonal
    ax.set_xlim(xlims[i])

# Add lines for off-diagonal plots (2D contours)
for i in range(n_params):
    for j in range(i):
        ax = axes[i * n_params + j]  # Lower triangular element
        ax.axvline(truths[j], color='k', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(truths[i], color='k', linestyle='--', linewidth=1, alpha=0.6)
        ax.scatter(
            truths[j], truths[i], 
            color='k', 
            marker='o', 
            s=1,
            alpha=0.8, 
            edgecolor='k', 
            linewidth=0.5,
            zorder=10
        )
        # Set xlim and ylim for off-diagonal
        ax.set_xlim(xlims[j])
        ax.set_ylim(xlims[i])

# add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='C0', lw=3, label='SBI Samples'),
    Line2D([0], [0], color='C5', lw=3, label='MCMC Samples'),
]
axes[1].legend(handles=legend_elements, loc='upper left', fontsize=20)

#
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# After you create your corner plot and before plt.savefig
# Add an inset axes in the upper right corner of the figure
inset_ax = inset_axes(
    axes[0],  # Use the first axes as the parent
    width="150%",
    height="150%",
    loc='upper right',
    borderpad=2,
    bbox_to_anchor=(4.1, 0., 1.0, 1.0),  # (x0, y0, width, height) in axes fraction
    bbox_transform=axes[0].transAxes,
)

# Example: plot a histogram of the first parameter in the inset
# inset_ax.hist(samples[:, 0], bins=20, color='C0', alpha=0.7)
# Inset: 2D corner plot using corner.corner
x = samp_mcmc[:, 0]
y = samp_mcmc[:, 1]

ff = corner.hist2d(
    x, y,
    ax=inset_ax,
    plot_contours=True,
    fill_contours=True,
    plot_datapoints=False,
    color='C5',
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    contour_kwargs={"colors": "C5"},
    bins=30,
    smooth=1.0,
)
plt.axvline(truths[0], color='k', linestyle='--', linewidth=1, alpha=0.8)
plt.axhline(truths[1], color='k', linestyle='--', linewidth=1, alpha=0.8)
from pastamarkers import pasta
plt.scatter(truths[0], truths[1], color='gold', marker=pasta.tortellini, s=100, edgecolors='k',zorder=10, alpha=0.8, linewidths=0.1)

plt.xlim(x.min()*0.999, x.max()*1.001)
plt.ylim(y.min()*0.99, y.max()*1.01)
for spine in plt.gca().spines.values():
    spine.set_edgecolor('C2')
    spine.set_linewidth(3)
    # set dashed border
    spine.set_linestyle('--')

inset_ax.set_title("Zoom in", fontsize=16)
inset_ax.set_xlabel(labels[0])
inset_ax.set_ylabel(labels[1])
# Get the limits of the inset
x0, x1 = x.min()*0.999, x.max()*1.001
y0, y1 = y.min()*0.99, y.max()*1.01

# Find the correct axes for (log10(m1), m2) -- usually axes[1] in corner's get_axes()
main_ax = axes[5]  # log10(m1) vs m2
import matplotlib.patches as patches

# Draw a rectangle to show the zoomed region
rect = patches.Rectangle(
    (x0, y0), x1 - x0, y1 - y0,
    linewidth=1.5, edgecolor='C2', facecolor='none', linestyle='--', zorder=20
)
main_ax.add_patch(rect)

# from matplotlib.patches import ConnectionPatch

# # Get the bounds of the inset axes in axes fraction coordinates
# inset_bbox = inset_ax.get_position()
# fig = inset_ax.figure

# for (main_xy, inset_xy) in [
#     # ((x0, y0), (0, 0)),   # lower left
#     ((x1, y1), (1, 1)),   # upper right
#     ((x1, y0), (1, 0)),   # lower right
#     # ((x0, y1), (0, 1)),   # upper left
# ]:
#     con = ConnectionPatch(
#         xyA=inset_xy, coordsA=inset_ax.transAxes,
#         xyB=main_xy, coordsB=main_ax.transData,
#         color="C2", linewidth=1.5, linestyle='--', zorder=100
#     )
#     fig.add_artist(con)
# Save figure
output_file = 'corner_plot_round_final.pdf'
print(f"Saving plot to {output_file}")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
# plt.show()
plt.close()

# Print parameter estimates (median and 68% credible intervals)
print("\nParameter estimates:")
param_labels = ['log10(m1)', 'm2', 'a', 'Tpl', 'ef']

for i, label in enumerate(param_labels):
    q16, q50, q84 = np.percentile(samp_mcmc[:, i], [16, 50, 84])
    err_plus = q84 - q50
    err_minus = q50 - q16
    rel_prec = (err_plus + err_minus) / (2 * abs(q50)) if q50 != 0 else np.nan
    if i == 0:  # log10(m1) - also show m1
        print(f"{label}: {q50:.3f} +{err_plus:.3f} -{err_minus:.3f} (rel. prec: {rel_prec:.3%})")
        print(f"  True value: {truths[i]:.3f}")
        print(f"  Equivalent m1: {10**q50:.2e} +{10**q84-10**q50:.2e} -{10**q50-10**q16:.2e} [M_sun]")
    elif i == 1:  # m2 - use scientific notation
        print(f"{label}: {q50:.2e} +{err_plus:.2e} -{err_minus:.2e} (rel. prec: {rel_prec:.3%})")
        print(f"  True value: {truths[i]:.2e}")
    else:  # a, Tpl, ef - use regular notation
        print(f"{label}: {q50:.4f} +{err_plus:.4f} -{err_minus:.4f} (rel. prec: {rel_prec:.3%})")
        print(f"  True value: {truths[i]:.4f}")

print(f"\nCorner plot saved to: {output_file}")

import matplotlib.colors as mcolors

# Select top likelihood samples (as before)
# mask_top = sample_lls > sample_lls.max() - 200
mask_top = np.argsort(sample_lls)[-5000:]  # Top 1000 samples

# Concatenate for a unique scatter plot
ind_0, ind_1 = 0, 1  # indices for log10(m1) and m2
xi = samples[mask_top, ind_0]
yi = samples[mask_top, ind_1]
zi = sample_lls[mask_top]

norm = mcolors.Normalize(vmin=zi.min(), vmax=zi.max())
cmap = plt.cm.viridis

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))
sc = ax.scatter(xi, yi, zi, marker='.', c=cmap(norm(zi)), s=100, alpha=0.8)

ind = np.argmax(loglike)
x0, y0, z0 = samp_mcmc[ind, ind_0], samp_mcmc[ind, ind_1], loglike.max()

# highlight the point
sc2 = ax.scatter(x0, y0, z0, marker='*', c='red', s=300, edgecolors='k')

# draw a vertical line from the bottom of the cloud up to the point
zmin = zi.min()  # or use ax.get_zlim()[0] after plotting if you prefer
ax.plot([x0, x0], [y0, y0], [zmin, z0],color='k', linestyle='--', linewidth=1.5, zorder=5)
ax.set_xlabel(labels[ind_0], labelpad=20)
ax.set_ylabel(labels[ind_1], labelpad=20)
ax.set_zlabel('Detection Statistic', labelpad=30)

# Tilt z-axis tick labels and move them further from the axis
for label in ax.get_zticklabels():
    label.set_rotation(45)  # Tilt the labels
    label.set_verticalalignment('bottom')
    label.set_horizontalalignment('left')
    label.set_gid(15)  # Move further from axis (requires matplotlib >=3.4)

plt.tight_layout()
plt.savefig('3d_scatter_likelihood.png', dpi=300)
# plt.show()
print("difference between max likelihood and SBRI:", loglike.max() - sample_lls.max(), loglike.max(), sample_lls.max())
best_mcmc = samp_mcmc[ind]
best_sample_lls = samples[np.argmax(sample_lls)]

def rel_diff_percent(est, true):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (est - true) / true * 100.0

labels_short = ['log10(m1)', 'm2', 'a', 'Tpl', 'ef']

rel_mcmc = rel_diff_percent(best_mcmc, truths)
rel_samples = rel_diff_percent(best_sample_lls, truths)

print("Parameters at max likelihood (MCMC):", best_mcmc)
print("Relative differences vs truth (percent):")
for i, lab in enumerate(labels_short):
    rd = rel_mcmc[i]
    print(f"  {lab}: est={best_mcmc[i]:.6g}, true={truths[i]:.6g}, rel_diff={rd:.6f}%")

print("\nParameters at max sample_lls (SBI samples):", best_sample_lls)
print("Relative differences vs truth (percent):")
for i, lab in enumerate(labels_short):
    rd = rel_samples[i]
    print(f"  {lab}: est={best_sample_lls[i]:.6g}, true={truths[i]:.6g}, rel_diff={rd:.6f}%")

print(truths)