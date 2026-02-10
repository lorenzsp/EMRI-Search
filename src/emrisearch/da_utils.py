import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_smoothing_spline
from scipy.signal.windows import tukey
from few.utils.constants import YRSID_SI
from emri_utils import create_signal
from scipy.signal import welch

tdi_flag = True  # Set to True to use TDI PSD

ff, Sn = np.loadtxt('./LPA.txt',skiprows=1).T
cs = CubicSpline(ff, Sn**2)

ff_tdi, Sn_tdi = np.load('./TDI2_AE_psd.npy').T
Sn_tdi = np.clip(Sn_tdi, Sn_tdi[0], None)
cs_tdi = CubicSpline(ff_tdi, Sn_tdi)

if tdi_flag:
    cs_eval = cs_tdi
    print("Using TDI PSD")
else:
    cs_eval = cs
    print("Using LISA PSD")

# Define the PSD function
def psd(frequencies):
    ret = cs_eval(np.abs(frequencies))
    mask = (np.abs(frequencies)<1e-4)
    ret[mask] = cs_eval(1e-4)
    return ret


def compute_sfts(data, deltaT, wind, samples_per_sft):
    """
    Compute SFTs for the given data.

    Args:
        data (np.ndarray): Input time-domain data.
        deltaT (float): Time step in seconds.
        wind (np.ndarray): Window function of length samples_per_sft.
        samples_per_sft (int): Number of samples per SFT.

    Returns:
        np.ndarray: SFTs with shape (num_freq_bins, num_sfts).
    """
    num_sfts = data.size // samples_per_sft
    sfts = (
        deltaT
        * np.fft.rfft(
            wind * data[: num_sfts * samples_per_sft].reshape(-1, samples_per_sft), axis=1
        ).T
    )
    return sfts


def sft_inner_product(sft1, sft2, freq):
    """
    Compute the inner product of two SFTs in the frequency domain.

    Args:
        sft1 (np.ndarray): First SFT.
        sft2 (np.ndarray): Second SFT.
        freq (np.ndarray): Frequency bins corresponding to the SFTs.

    Returns:
        np.ndarray: Inner product of the two SFTs.
    """
    df = freq[1] - freq[0]
    return 4 * np.sum(sft1.conj() * sft2 / psd(freq)[:, None] * df, axis=0).real


def generate_noise(signal_length, deltaT, psd_func):
    """
    Generate Gaussian noise in the time domain with a given PSD.

    Args:
        signal_length (int): Length of the time-domain signal.
        deltaT (float): Time step in seconds.
        psd_func (callable): Function that returns PSD values for given frequencies.

    Returns:
        np.ndarray: Time-domain noise realization.
    """
    frequencies = np.fft.rfftfreq(signal_length, deltaT)
    sigma2 = psd_func(frequencies) * signal_length * deltaT / 4
    n_tilde = (
        np.random.normal(0.0, sigma2**0.5, size=len(frequencies))
        + 1j * np.random.normal(0.0, sigma2**0.5, size=len(frequencies))
    )
    noise_real = np.fft.irfft(n_tilde, n=signal_length) / deltaT
    return noise_real

def estimate_psd_with_welch(data, deltaT, window_length=604800):
    """
    Estimate the one-sided Power Spectral Density (PSD) using Welch's method.

    Args:
        data (np.ndarray): Input time-domain data.
        deltaT (float): Time step in seconds.
        window_length (float): Length of each segment in seconds.

    Returns
    -------
    psd_spline : scipy.interpolate.CubicSpline
        A callable spline that gives the estimated PSD as a function of frequency.
    """
    nperseg = int(window_length / deltaT)
    f, Pxx = welch(data, fs=1/deltaT, nperseg=nperseg, average='median')

    # Log-spaced smoothing: for each frequency, average over a window that spans a fixed factor in log-frequency
    log_factor = 1.1  # e.g., ±5% in frequency
    smooth_power = np.zeros_like(Pxx)
    for i, freq in enumerate(f):
        if freq == 0:
            smooth_power[i] = Pxx[i]
            continue
        # Find indices within [freq/log_factor, freq*log_factor]
        idx = np.where((f >= freq / log_factor) & (f <= freq * log_factor))[0]
        smooth_power[i] = np.median(Pxx[idx]) if len(idx) > 0 else Pxx[i]

    psd_spline = CubicSpline(f, smooth_power)
    return psd_spline

def estimate_psd_from_sfts(data_sfts, freq, window_size=500):
    """
    Estimate the one-sided Power Spectral Density (PSD) from Short Fourier Transforms (SFTs).

    Parameters
    ----------
    data_sfts : np.ndarray
        Array of shape (Nsfts, Nfreqs), complex SFT data for each time segment.
    freq : np.ndarray
        Array of frequency bins corresponding to the columns in `data_sfts`.
    window_size : int, optional
        Size of the moving average window for smoothing the estimated power (default: 500 bins).

    Returns
    -------
    psd_spline : scipy.interpolate.CubicSpline
        A callable spline that gives the estimated PSD as a function of frequency.
    """
    # Compute mean power spectrum across all SFTs, excluding edge bins
    # Exclude first and last frequency bin to avoid boundary artifacts
    power_spectrum = np.mean(np.abs(data_sfts[:, 1:-1])**2, axis=1)

    # Frequency bins used (excluding edges)
    freq_trimmed = freq[1:-1]

    # Smooth the power spectrum using a moving average
    smooth_power = np.convolve(
        power_spectrum, np.ones(window_size) / window_size, mode='same'
    )

    # Convert to one-sided PSD (factor of 2), scaled by frequency resolution
    df = np.diff(freq)[0]  # assumes uniform frequency spacing
    mean_psd = smooth_power * 2 * df
    
    # Interpolate PSD using cubic spline
    psd_spline = CubicSpline(freq_trimmed, mean_psd[1:-1])

    return psd_spline

def get_snr(signal, deltaT):
    ft_freq = np.fft.rfftfreq(len(signal), deltaT)
    SNR2_source = (np.abs(np.fft.rfft(signal)*deltaT)**2 / psd(np.abs(ft_freq))).sum() * 4 * np.diff(ft_freq)[0]
    return SNR2_source**0.5

def get_sfts(true_values, snr_ref, T_sft, T=1.0, deltaT=5.0):
    """
    Generates Short Fourier Transforms (SFTs) for a simulated EMRI signal and corresponding noise.

    The function creates an EMRI signal using the provided true parameter values, rescales the signal to a desired 
    signal-to-noise ratio (SNR), and computes SFTs for both the signal and simulated noise. The SFTs are computed 
    using a Tukey window and returned along with the corresponding time stamps.

    Args:
        true_values (array-like): Parameters for generating the EMRI signal.
        snr_ref (float): Reference SNR to which the signal should be rescaled.
        T_sft (float): Duration of each SFT segment.
        T (float, optional): Total duration of the signal (default is 1.0).
        deltaT (float, optional): Time step between samples (default is 5.0).

    Returns:
        t_alpha (np.ndarray): Array of time stamps corresponding to the start of each SFT segment.
        noise_sfts (np.ndarray): Array of SFTs computed from simulated noise.
        signal_sfts (np.ndarray): Array of SFTs computed from the rescaled EMRI signal.
    """
    # EMRI signal
    t, hp, hx = create_signal(true_values, T=T, deltaT=deltaT)
    snr_sig = get_snr(hp, deltaT)  # Check SNR of the signal
    hp, hx = snr_ref/snr_sig * hp, snr_ref/snr_sig * hx  # Rescale signal to desired SNR
    snr_sig = get_snr(hp, deltaT)  # Check SNR of the signal

    samples_per_sft = int(T_sft/deltaT)
    wind = tukey(samples_per_sft, 0.2)
    signal_sfts = compute_sfts(hp, deltaT, wind, samples_per_sft)
    num_sfts = signal_sfts.shape[1]
    t_alpha = np.arange(num_sfts) * T_sft

    noise_sfts = compute_sfts(generate_noise(hp.shape[0], deltaT, psd), deltaT, wind, samples_per_sft)
    return t_alpha, noise_sfts, signal_sfts

if __name__ == "__main__":
    from emri_utils import create_signal, get_f_fdot_fddot_back
    import matplotlib.pyplot as plt

    # test signal
    true_values = np.asarray([1e6, 10.0, 0.9, 0.7, 0.1, 1.0])
    deltaT = 5.0
    t, hp, hx = create_signal(true_values, T=1.0, deltaT=deltaT)
    samples_per_sft = int(5e4/deltaT)  # Example number of samples per SFT
    wind = tukey(samples_per_sft, 0.2)  # Example window function
    frequencies = np.fft.rfftfreq(samples_per_sft, deltaT)
    sfts = compute_sfts(hp, deltaT, wind, samples_per_sft)
    snr2 = sft_inner_product(sfts, sfts, frequencies)
    t_alpha = np.arange(sfts.shape[1]) * 5e4  # Example time stamps for SFTs
    true_phi_f_fdot_fddot = get_f_fdot_fddot_back(true_values, t_alpha)
    
    plt.figure()
    plt.title(f"SNR = {snr2.sum()**0.5}")
    plt.imshow(
        np.abs(sfts),
        aspect='auto',
        origin='lower',
        extent=[0, t[-1], frequencies[0], frequencies[-1]],
        cmap='viridis'
    )
    plt.plot(t_alpha, 2 * true_phi_f_fdot_fddot[1][0], 'r:')
    plt.title('Spectrogram of the Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.yscale('log')
    plt.ylim(1e-4, 1e-2)
    plt.show()

    noise_real = generate_noise(hp.shape[0], deltaT, psd)
    noise_sfts = compute_sfts(noise_real, deltaT, wind, samples_per_sft)
    snr2 = sft_inner_product(noise_sfts, noise_sfts, frequencies)
    snr_noise = snr2.sum() / hp.shape[0]
    print(snr_noise, "approx 1, not perfectly due to windowing") #TODO Check Find chirp from Allen to fix normalization
    # Test noise standard inner product
    snr_noise = get_snr(noise_real, deltaT)**2 / hp.shape[0]
    print(snr_noise, "approx 1")

    # estimate PSD
    psd_spline_from_data = estimate_psd_with_welch(hp + noise_real, deltaT)
    psd_spline_from_noise = estimate_psd_with_welch(noise_real, deltaT)
    
    # plot PSD
    plt.figure()
    plt.title("Power Spectral Density (PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (1/Hz)")
    plt.loglog(frequencies, psd(frequencies), label="True PSD")
    plt.loglog(frequencies, psd_spline_from_data(frequencies), '--', label="Estimated PSD (Data)")
    plt.loglog(frequencies, psd_spline_from_noise(frequencies), '--', label="Estimated PSD (Noise)")
    plt.legend()
    plt.show()