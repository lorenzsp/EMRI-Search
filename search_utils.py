import numpy as np
from scipy.special import fresnel
from da_utils import psd, sft_inner_product, compute_sfts, generate_noise, get_snr
from scipy.signal.windows import tukey
from emri_utils import create_signal, get_f_fdot_fddot_back
from few.utils.constants import YRSID_SI

def dirichlet_kernel(f_0, T_sft):
    """The physically correct kernel for the f_1 = 0 case."""
    return T_sft * np.exp(1j * np.pi * f_0 * T_sft) * np.sinc(f_0 * T_sft)

def fresnel_kernel(f_0, f_1, T_sft):
    """
    Fresnel kernel.

    Args:
        f_0 (float or np.ndarray): Initial frequency.
        f_1 (float or np.ndarray): Frequency derivative.
        T_sft (float): Duration of the SFT segment.

    Returns:
        np.ndarray: Fresnel kernel values.
    """
    quot = f_0 / f_1
    factor = np.sqrt(2 * f_1)

    Sl, Cl = fresnel(factor * quot)
    Su, Cu = fresnel(factor * (quot + T_sft))

    return np.exp(-1j * np.pi * f_0**2 / f_1) * ((Cu - Cl) + 1j * (Su - Sl)) / factor

def robust_kernel(f_0, f_1, T_sft):
    """
    This is the function we will call. It computes the forward pass.
    """
    abs_f_1 = np.abs(f_1)
    return np.where(
        abs_f_1 > 0., 
        np.where(
            f_1 > 0,
            fresnel_kernel(f_0, abs_f_1, T_sft),
            fresnel_kernel(-f_0, abs_f_1, T_sft).conj(),
        ),
        dirichlet_kernel(f_0, T_sft),
    )

def det_stat(data_sfts, A_alpha, phi_alpha, f_alpha, fdot_alpha, P=100, T_sft=86400.):
    """
    Compute the detection statistic for a given set of parameters and data.
    This function calculates a detection statistic based on the input Short Fourier Transforms (SFTs) 
    and signal parameters. It evaluates the likelihood of a signal being present in the data.
    Args:
        data_sfts (jax.numpy.ndarray): The input SFT data, typically a 2D array where the first 
            dimension corresponds to frequency bins and the second dimension corresponds to time segments.
        A_alpha (jax.numpy.ndarray): Amplitude parameters of the signal.
        phi_alpha (jax.numpy.ndarray): Phase parameters of the signal.
        f_alpha (jax.numpy.ndarray): Frequency parameters of the signal.
        fdot_alpha (jax.numpy.ndarray): Frequency derivative parameters of the signal.
        P (int, optional): The range of frequency bins to consider around the signal frequency. 
            Defaults to 100.
        T_sft (float, optional): The duration of each SFT segment in seconds. Defaults to 86400 (1 day).
    Returns:
        float: The detection statistic value, normalized by the square root of the noise-weighted 
        signal power.
    Notes:
        - The function uses Fresnel kernels to compute the signal contribution in the frequency domain.
        - Out-of-range frequency bins are masked to zero to avoid indexing errors.
        - The detection statistic is computed as the ratio of the matched-filter term (`dh_term`) 
          to the square root of the noise-weighted signal power (`hh_term`).
    """

    # Non-signal-dependent values are passed here by clousure
    deltaf = 1 / T_sft

    f_k_of_alpha = (f_alpha * T_sft).astype(int)
    k_min_max = f_k_of_alpha + np.arange(-P, P + 1)[:, None]

    # Set to 0 whatever gets beyond the range.
    # Note that jax will not complain about out-of-range indexing
    zero_mask = (k_min_max >= 0) & (k_min_max < data_sfts.shape[0])

    c_alpha = (
        deltaf
        * data_sfts[k_min_max, np.arange(data_sfts.shape[1])].conj() / psd(k_min_max * deltaf)
        * robust_kernel(f_alpha - k_min_max * deltaf, fdot_alpha, T_sft)
        * zero_mask
    ).sum(axis=0)

    c_alpha = np.nan_to_num(c_alpha, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    dh_term = 2 * A_alpha * c_alpha * np.exp(1j * phi_alpha)
    hh_term = T_sft * (A_alpha**2/psd(f_alpha))/2
    return dh_term, hh_term

def generate_emri_signal_and_sfts(true_values, T_data, T_sft, deltaT, snr_ref, T_snr):
    """
    Generate EMRI signal, compute SFTs, and prepare data for analysis.
    
    Parameters:
    -----------
    true_values : np.ndarray
        EMRI parameters [m1, m2, a, Tpl, ef, x0]
    T_data : float
        Total data duration in years
    T_sft : float
        SFT duration in seconds
    deltaT : float
        Time step in seconds
    snr_ref : float
        SNR of the signal with duration T_snr
    T_snr : float
        Duration of the signal in years it must be always bigger or equal to T_data

    Returns:
    --------
    dict : Dictionary containing:
        - 't': trimmed time array
        - 'signal_sfts': SFTs of the signal
        - 'noise_sfts': SFTs of noise realization
        - 'new_noise_sfts': SFTs of second noise realization
        - 'data_sfts': SFTs of signal + noise
        - 't_alpha': time samples for SFTs
        - 'samples_per_sft': number of samples per SFT
        - 'wind': window function
        - 'snr_final': final SNR of scaled signal
        - 'true_phi_f_fdot_fddot': frequency evolution parameters
    """
    # Generate EMRI signal
    t, hp, hx, param_dict = create_signal(true_values, T=T_snr, deltaT=deltaT, return_params=True, randomize=True)
    snr_sig = get_snr(hp, deltaT)  # Check SNR of the signal
    
    # Trim signal to desired duration
    mask = t < YRSID_SI * T_data
    hp, hx = snr_ref/snr_sig * hp[mask], snr_ref/snr_sig * hx[mask]  # Rescale signal to desired SNR
    t = t[mask]
    snr_final = get_snr(hp, deltaT)  
    print(f"Final SNR: {snr_final}")
    # update distance
    param_dict['dist'] = param_dict['dist'] * snr_sig/snr_ref
    
    # SFT parameters
    samples_per_sft = int(T_sft/deltaT) 
    wind = tukey(samples_per_sft, 0.2)  # Window function 
    
    # Compute SFTs
    signal_sfts = compute_sfts(hp, deltaT, wind, samples_per_sft)
    num_sfts = signal_sfts.shape[1]
    t_alpha = np.arange(num_sfts) * T_sft
    
    # Generate noise SFTs
    noise_sfts = compute_sfts(generate_noise(hp.shape[0], deltaT, psd), deltaT, wind, samples_per_sft)
    new_noise_sfts = compute_sfts(generate_noise(hp.shape[0], deltaT, psd), deltaT, wind, samples_per_sft)
    data_sfts = signal_sfts + noise_sfts
    
    # compute matched filtering
    freq = np.fft.rfftfreq(samples_per_sft, deltaT)
    m_hd = sft_inner_product(signal_sfts, data_sfts, freq)
    m_hh = sft_inner_product(signal_sfts, signal_sfts, freq)
    m_hn = sft_inner_product(signal_sfts, noise_sfts, freq)

    # Get frequency evolution for true values
    true_phi_f_fdot_fddot = get_f_fdot_fddot_back(true_values, t_alpha)
    
    return {
        'param_dict': param_dict,
        't': t,
        'data_sfts': data_sfts,
        'signal_sfts': signal_sfts,
        'noise_sfts': noise_sfts,
        'new_noise_sfts': new_noise_sfts,
        't_alpha': t_alpha,
        'samples_per_sft': samples_per_sft,
        'num_sfts': num_sfts,
        'wind': wind,
        'snr_final': snr_final,
        'true_phi_f_fdot_fddot': true_phi_f_fdot_fddot,
        'm_hd': m_hd,
        'm_hh': m_hh,
        'm_hn': m_hn
    }

if __name__ == "__main__":
    from few.utils.constants import YRSID_SI
    from matplotlib import pyplot as plt
    T_data = 0.2  # in years
    T_snr = 0.2  # in years
    deltaT = 5.0 # in seconds
    T_sft = 5e4
    Tpl = 0.2 # test shorter
    # Tpl = 0.15 # test longer
    true_values = np.asarray([1e6, 10.0, 0.1, Tpl, 0.1, 1.0])

    # EMRI signal
    snr_ref = 30.0  # Desired SNR of the signal
    injection_dict = generate_emri_signal_and_sfts(true_values, T_data, T_sft, deltaT, snr_ref, T_snr)
    samples_per_sft = injection_dict['samples_per_sft']
    num_sfts = injection_dict['num_sfts']
    ############################################
    # standard matched snr
    m_hd = injection_dict['m_hd']
    m_hh = injection_dict['m_hh']
    m_opt_snr = m_hh**0.5 
    m_snr = m_hd / m_opt_snr
    m_hn = injection_dict['m_hn']
    m_snr_hn = m_hn / m_opt_snr
    #############################################
    # new stat
    m=2
    n=0
    # EMRI traj
    t_alpha = injection_dict['t_alpha']
    phi, f, dotf, dotdotf = injection_dict['true_phi_f_fdot_fddot']
    # Compute detection statistics information
    phi_alpha = m*phi[0] + n*phi[1]
    f_alpha = m*f[0] + n*f[1]
    fdot_alpha = m*dotf[0] + n*dotf[1]
    fddot_alpha = m*dotdotf[0] + n*dotdotf[1]
    delta_phi_approx = np.abs(fddot_alpha * T_sft**3 / 6)
    # & (delta_phi_approx < 0.1)
    A_alpha = np.where((f_alpha > 5e-4) & (fdot_alpha > 0), 1.0, 0.0)
    dh, hh = det_stat(injection_dict['data_sfts'], A_alpha, phi_alpha, f_alpha, fdot_alpha, T_sft=T_sft)
    ratio = np.abs(dh)**2 / hh
    nh, hh = det_stat(injection_dict['noise_sfts'], A_alpha, phi_alpha, f_alpha, fdot_alpha, T_sft=T_sft)
    ratio_n = np.abs(nh)**2 / hh
    s_hh, hh = det_stat(injection_dict['signal_sfts'], A_alpha, phi_alpha, f_alpha, fdot_alpha, T_sft=T_sft)
    ratio_s = np.abs(s_hh)**2 / hh
    print('Next values can be checked against jax utils\n')
    print(f"Log-likelihood ratio: {0.5*np.nan_to_num(ratio).sum()}, {0.5*np.nan_to_num(ratio_n).sum()}, {0.5*np.nan_to_num(ratio_s).sum()} \n")
    ############################################
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    t_alpha = t_alpha/86400  # Convert to days
    # First subplot: SNR per segment
    ax1.plot(t_alpha, m_opt_snr.T, "-",  color='C0', label="<h|h>")
    ax1.plot(t_alpha, m_snr.T, "--",  color='C1', label="<d|h>")
    ax1.plot(t_alpha, m_snr_hn.T, ":",  color='C2', label="<n|h>")
    ax1.plot(t_alpha, ratio_s**0.5, "-P",  color='C0', label=r"$S_\alpha(h)$")
    ax1.plot(t_alpha, ratio**0.5, "-x",  color='C1', label=r"$S_\alpha(d)$")
    ax1.plot(t_alpha, ratio_n**0.5, "-o",  color='C2', label=r"$S_\alpha(n)$")
    ax1.set_ylabel("Statistic per segment")
    # ax1.legend()
    ax1.grid()
    # Second subplot: Cumulative SNR
    ax2.plot(t_alpha, np.cumsum(m_opt_snr**2)**0.5, "-", color='C0', label="<h|h>")
    ax2.plot(t_alpha, np.cumsum(m_snr**2)**0.5, "--", color='C1', label="<d|h>")
    ax2.plot(t_alpha, np.cumsum(m_snr_hn**2)**0.5, ":", color='C2', label="<n|h>")
    ax2.plot(t_alpha, np.cumsum(ratio_s)**0.5, "-P",  color='C0', label=r"$S_\alpha(h)$")
    ax2.plot(t_alpha, np.cumsum(ratio)**0.5, "-x",  color='C1', label=r"$S_\alpha(d)$")
    ax2.plot(t_alpha, np.cumsum(ratio_n)**0.5, "-o",  color='C2', label=r"$S_\alpha(n)$")
    ax2.legend()
    ax2.set_xlabel("Time [days]")
    ax2.set_ylabel("Cumulative Statistic")
    ax2.grid()

    plt.tight_layout()
    plt.savefig("det_stat_example.png")
    plt.show()
############################################
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 12))
    t_alpha_days = t_alpha / 86400  # Convert to days

    # First subplot: SNR per segment
    ax1.plot(t_alpha_days, m_opt_snr.T, "-",  color='C0', label="<h|h>")
    ax1.plot(t_alpha_days, m_snr.T, "--",  color='C1', label="<d|h>")
    ax1.plot(t_alpha_days, m_snr_hn.T, ":",  color='C2', label="<n|h>")
    ax1.plot(t_alpha_days, ratio_s**0.5, "-P",  color='C0', label=r"$S_\alpha(h)$")
    ax1.plot(t_alpha_days, ratio**0.5, "-x",  color='C1', label=r"$S_\alpha(d)$")
    ax1.plot(t_alpha_days, ratio_n**0.5, "-o",  color='C2', label=r"$S_\alpha(n)$")
    ax1.set_ylabel("Statistic per segment")
    ax1.grid()

    # Second subplot: Cumulative SNR
    ax2.plot(t_alpha_days, np.cumsum(m_opt_snr**2)**0.5, "-", color='C0', label="<h|h>")
    ax2.plot(t_alpha_days, np.cumsum(m_snr**2)**0.5, "--", color='C1', label="<d|h>")
    ax2.plot(t_alpha_days, np.cumsum(m_snr_hn**2)**0.5, ":", color='C2', label="<n|h>")
    ax2.plot(t_alpha_days, np.cumsum(ratio_s)**0.5, "-P",  color='C0', label=r"$S_\alpha(h)$")
    ax2.plot(t_alpha_days, np.cumsum(ratio)**0.5, "-x",  color='C1', label=r"$S_\alpha(d)$")
    ax2.plot(t_alpha_days, np.cumsum(ratio_n)**0.5, "-o",  color='C2', label=r"$S_\alpha(n)$")
    ax2.legend()
    ax2.set_xlabel("Time [days]")
    ax2.set_ylabel("Cumulative Statistic")
    ax2.grid()

    # Third subplot: Derivative of SNR per segment
    snr_arrays = [
        # ("<h|h>", m_opt_snr.T, 'C0'),
        # ("<d|h>", m_snr.T, 'C1'),
        # ("<n|h>", m_snr_hn.T, 'C2'),
        (r"$S_\alpha(h)$", ratio_s**0.5, 'C0'),
        (r"$S_\alpha(d)$", ratio**0.5, 'C1'),
        (r"$S_\alpha(n)$", ratio_n**0.5, 'C2'),
    ]
    for label, arr, color in snr_arrays:
        # Use np.gradient for derivative, divide by time step in days
        dt_days = np.mean(np.diff(t_alpha_days))
        snr_deriv = np.gradient(arr, dt_days)
        ax3.plot(t_alpha_days, snr_deriv, label=f"d/dt {label}", color=color)
    ax3.set_ylabel("d(SNR)/dt [per day]")
    ax3.set_xlabel("Time [days]")
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    plt.savefig("det_stat_example.png")
    plt.show()

    # estimate of the noise
    mean_psd = np.mean(np.abs(injection_dict['data_sfts'])**2,axis=1)
    ############################################
    # investigate Chi2 stats
    det_stat_n = []
    chi2_n = []
    for real_i in range(100):
        noise = generate_noise(int(YRSID_SI*T_data/deltaT), deltaT, psd)
        noise_sfts = compute_sfts(noise, deltaT, injection_dict['wind'], samples_per_sft)
        nh, hh = det_stat(noise_sfts, A_alpha, phi_alpha, f_alpha, fdot_alpha, T_sft=T_sft)
        ratio_n = np.abs(nh)**2 / hh
        det_stat_n.append(np.nan_to_num(ratio_n).sum())

        term1_n, term2_n = np.nan_to_num(np.abs(nh)**2 / hh).sum() , len(hh) * np.sum(np.abs(nh)**2) / np.sum(hh)
        chi2_n.append(term2_n-term1_n)

    for el,name in zip([chi2_n, det_stat_n], ['Chi2 Noise', 'Detection Statistic Noise']):
        plt.figure()
        plt.hist(el, bins=30, alpha=0.7)
        # plt.axvline(len(hh)/2)
        # plt.axvline((noise_sfts.shape[0] * 8) **0.5)
        plt.xlabel(name)
        plt.grid()
        plt.show()


