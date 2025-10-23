import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfcinv
from scipy.stats import rice
from scipy.stats import rayleigh
import h5py
from scipy.stats import gumbel_r


# --- Constants from the paper ---
MU_K = 2.00
SIGMA_K = 1.45

def get_snr2_ev_fact(N):
    snr2_ev_fact = np.arange(1, N+1)
    snr2_ev_fact = snr2_ev_fact/snr2_ev_fact.sum()
    return snr2_ev_fact
    
def detection_threshold(P_F, mu0, sigma0):
    """Eq. (70): Detection threshold Υ*(P_F)"""
    return mu0 + np.sqrt(2) * sigma0 * erfcinv(2 * P_F)

def detection_probability(A, N, P_F):
    """Eq. (71): Detection probability for given A, N"""
    mu0 = N * MU_K
    sigma0 = np.sqrt(2 * N * SIGMA_K**2)
    Y_thresh = detection_threshold(P_F, mu0, sigma0)
    mu1 = N * MU_K + A**2
    sigma1 = np.sqrt(2 * N * SIGMA_K**2 + 4 * A**2)
    pdet = 0.5 * (1 + erf((mu1 - Y_thresh) / (np.sqrt(2) * sigma1)))
    return pdet

def log_detection_probability_rice(A, P_F):
    """Detection probability using the Rice distribution."""
    rho_star = np.sqrt(-2*np.log(P_F)) # missing from eq 64 of https://arxiv.org/pdf/1705.04259
    # Survival function (also defined as 1 - cdf)
    # int rho infinity p(rho|A) drho = 1-cdf(rho|A)
    out = rice.logsf(rho_star, b=A)
    return out

def compute_detection_data(N_values, A_values, P_F_tot, N_templates, PD_val=0.9):
    """Compute detection probabilities and critical amplitudes."""
    P_F = P_F_tot / N_templates
    Ac_values = []
    PD_curves = []
    PD_match_filter = np.exp(np.array([log_detection_probability_rice(A, P_F) for A in A_values]))

    for N in N_values:
        if N == 1:
            PD_curves.append(PD_match_filter)
            Ac = np.interp(PD_val, PD_match_filter, A_values)
            Ac_values.append(Ac)
        else:
            PD = np.array([detection_probability(A, N, P_F) for A in A_values])
            PD_curves.append(PD)
            Ac = np.interp(PD_val, PD, A_values)
            Ac_values.append(Ac)
    
    return Ac_values, PD_curves, PD_match_filter

def plot_detection_analysis(N_values, A_values, Ac_values, PD_curves, PD_match_filter):
    """Create the detection analysis plots."""
    colors = ['#1f77b4', '#8c564b', '#d95f02', '#e6ab02']
    
    # Theoretical power law (Eq. 72)
    N_fit = np.linspace(1, 1700, 300)
    Ac_fit = 6.57 * N_fit**0.235
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))
    
    # Top subplot: PD(A) curves
    for i, (color, N) in enumerate(zip(colors, N_values)):
        PD = PD_curves[i]
        Ac = Ac_values[i]
        
        ax1.plot(A_values, PD, lw=2.5, color=color, label=f"N = {N}")
        ax1.axvline(Ac, color=color, ls="--", alpha=0.6)
        ax1.plot(Ac, 0.9, "o", color=color)
    # plot match filter curve
    ax1.plot(A_values, PD_match_filter, 'k-', lw=2, label='Match filter (Rice)', zorder=2)
    ax1.set_xlabel("Signal amplitude $A$")
    ax1.set_ylabel("Detection probability $P_D$")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 50)
    ax1.legend()
    ax1.set_title("Detection probability $P_D(A)$ for different $N$")
    ax1.grid(alpha=0.3)
    ax1.axhline(0.9, color='gray', ls='--', alpha=0.6)
    
    # Bottom subplot: Ac(N)
    ax2.plot(N_fit, Ac_fit, 'k-', lw=2, label=r'Eq. (72): $A_c \approx 6.57 N^{0.235}$')
    ax2.scatter(N_values, Ac_values, c=colors[:len(N_values)], zorder=3)
    ax2.set_xlabel("Number of segments $N$")
    ax2.set_ylabel("Threshold amplitude $A_c$")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1700)
    ax2.legend()
    ax2.set_title("Threshold amplitude $A_c$ vs number of segments $N$")
    
    plt.tight_layout()
    plt.show()

def get_maximum(N_realizations, N, A, standard_sum=False):
        
    # randomly draw signal time to plunge
    tpl = np.random.uniform(0.1, 1.0, size=N_realizations)
    
    # signal snr2 grows like n
    n_vec = np.repeat(np.arange(1, N+1)/(N), N_realizations).reshape(N, N_realizations)
    # cut the 
    mask = n_vec >= tpl
    # print("Mask fraction", np.mean(mask))

    # normalize
    if N != 1:
        n_vec[mask] = 0.0
        # n_vec[~mask] = 1.0
        n_vec = n_vec / n_vec.sum(axis=0)
    # print("check snr consistency",np.mean(np.sum(n_vec * A**2,axis=0)**0.5), A)
    
    # noise: Rayleigh
    rho2_per_segment_noise = rayleigh.rvs(size=(N,N_realizations))**2
    # signal: Rice
    rho2_per_segment_signal = rice.rvs(b=n_vec**0.5 * A, size=(N, N_realizations))**2
    
    # # plot evolution of the statistic
    # plt.figure(figsize=(10,6))
    # plt.plot(rho_noise[:,:10], color='red', alpha=0.2)
    # plt.plot(rho_signal[:,:10], color='blue', alpha=0.2)
    # plt.xlabel('Number of segments')
    # plt.ylabel('Detection statistic ρ')
    # plt.title(f'Evolution of detection statistic for N={N}, A={A}')
    # plt.show()
    if standard_sum:
        rho_noise = np.sum(rho2_per_segment_noise, axis=0)
        rho_signal = np.sum(rho2_per_segment_signal, axis=0)
        return rho_signal, rho_noise


    rho_noise = np.cumsum(rho2_per_segment_noise, axis=0)
    rho_signal = np.cumsum(rho2_per_segment_signal, axis=0)

    # mean across realizations
    mean_noise = rho_noise.mean(axis=1)
    std_noise = rho_noise.std(axis=1)

    # normalize
    rho_signal = (rho_signal - mean_noise[:, None]) / std_noise[:, None]
    rho_noise = (rho_noise - mean_noise[:, None]) / std_noise[:, None]
    
    # take the maximum over Tpl
    # rho_signal = rho_signal.max(axis=0)
    # rho_noise = rho_noise.max(axis=0)
    rho_signal = rho_signal.sum(axis=0)
    rho_noise = rho_noise.sum(axis=0)
    
    
    # take the last one
    # rho_signal = rho_signal[-1, :]
    # rho_noise = rho_noise[-1, :]

    # plt.figure(figsize=(10,6))
    # plt.hist(rho_noise, bins=50, density=True, alpha=0.5, label='Noise')
    # plt.hist(rho_signal, bins=50, density=True, alpha=0.5, label='Signal+Noise')
    # # plt.axvline( N * MU_K, color='blue', ls='--', label='Mean Noise')
    # # plt.axvline( N * MU_K + A**2, color='orange', ls='--', label='Mean Signal+Noise')
    # plt.xlabel('Detection statistic ρ')
    # plt.ylabel('Probability density')
    # plt.legend()
    # plt.title(f'Histograms of detection statistic for N={N}, A={A}')
    # plt.show()

    return rho_signal, rho_noise

def compute_detection_probability_simulation(N, P_F_tot, N_realizations, A_range, standard_sum=True):
    """
    Compute detection probability using Monte Carlo simulation.
    
    Args:
        N: Number of segments
        P_F_tot: False alarm probability
        N_realizations: Number of Monte Carlo realizations
        A_range: Array of amplitude values to test
        standard_sum: Whether to use standard sum (True) or custom statistic (False)
    
    Returns:
        List of detection probabilities for each amplitude
    """
    P_D_values = []
    rho_sig = []
    rho_noise = []
    for A_test in A_range:
        rho_sig_test, rho_noise_test = get_maximum(N_realizations, N, A_test, standard_sum=standard_sum)
        detection_threshold = np.quantile(rho_noise_test, 1 - P_F_tot)
        P_D = np.mean(rho_sig_test > detection_threshold)
        P_D_values.append(P_D)
        rho_sig.append(rho_sig_test)
        rho_noise.append(rho_noise_test)
    # transform into arrays
    rho_sig = np.array(rho_sig)
    rho_noise = np.array(rho_noise)
    P_D_values = np.array(P_D_values)
    return P_D_values, rho_sig, rho_noise

def main():
    """Main function to run the detection analysis."""
    P_F_tot = 1e-2/1e25
    N_templates = 1e40 * 1e8
    N_values = [1, 632]
    max_snr = 40
    A_values = np.linspace(1, max_snr, 5000)

    # Ac_values, PD_curves, PD_match_filter = compute_detection_data(N_values, A_values, P_F_tot, N_templates)
    # plot_detection_analysis(N_values, A_values, Ac_values, PD_curves, PD_match_filter)
    
    # Create detection probability plot
    N_realizations = 1_000
    # Theoretical curves
    PD_match_filter = np.exp(np.array([log_detection_probability_rice(A, P_F_tot) for A in A_values]))
    PD_curve = np.array([detection_probability(A, 632, P_F_tot) for A in A_values])
    
    # coherent case
    num_space = 50
    A_range = np.linspace(0, max_snr, num_space)
    P_D_values_mf, rho_sig_mf, rho_noise_mf = compute_detection_probability_simulation(1, P_F_tot, N_realizations, A_range, standard_sum=True)
    # semi-coherent case
    P_D_values, rho_sig_semi, rho_noise_semi = compute_detection_probability_simulation(632, P_F_tot, N_realizations, A_range, standard_sum=True)
    my_stat_values, rho_sig_my, rho_noise_my = compute_detection_probability_simulation(632, P_F_tot, N_realizations, A_range, standard_sum=False)

    plt.figure()
    plt.plot(A_values, PD_match_filter, '-', color='C0', lw=2, label='Theoretical coherent', zorder=2)
    plt.plot(A_range, P_D_values_mf, '--', color='C1',lw=2, label='Simulated coherent', zorder=2)
    plt.plot(A_values, PD_curve, '-', color='C2', linewidth=2, label='Theoretical semi-coherent')
    plt.plot(A_range, P_D_values, '--', color='C3', linewidth=2, label='Simulated semi-coherent')
    plt.plot(A_range, my_stat_values, '--', color='C4', linewidth=2, label='My simulated semi-coherent')
    plt.xlabel('SNR')
    plt.ylabel('Detection probability')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.xlim(0, 50)
    plt.tight_layout()
    plt.savefig('theoretical_detection_probability.png', dpi=300)
    plt.show()

    # # plot histograms
    # plt.figure()
    # plt.hist(rho_sig_mf[0], density=True, alpha=0.6, label='Simulated coherent')
    # plt.hist(rho_noise_mf.flatten(), density=True, alpha=0.6, label='Simulated coherent noise')
    # plt.hist(rho_sig_semi[-1], density=True, alpha=0.6, label='Simulated semi-coherent')
    # plt.hist(rho_noise_semi.flatten(), density=True, alpha=0.6, label='Simulated semi-coherent noise')
    # plt.hist(rho_sig_my[1], density=True, alpha=0.6, label='My simulated semi-coherent')
    # plt.hist(rho_noise_my.flatten(), density=True, alpha=0.6, label='My simulated semi-coherent noise')
    # plt.xlabel('Detection statistic ρ')
    # plt.ylabel('Probability density')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()