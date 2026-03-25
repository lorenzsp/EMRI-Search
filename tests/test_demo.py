import unittest

class TestDaUtils(unittest.TestCase):
    def test_dummy(self):
        from emrisearch.emri_utils import create_signal, get_f_fdot_fddot_back
        from emrisearch.da_utils import compute_sfts, sft_inner_product, generate_noise, get_snr, estimate_psd_with_welch, psd
        import jax.numpy as jnp
        import numpy as np
        from scipy.signal.windows import tukey
        from emrisearch.search_utils import generate_emri_signal_and_sfts
        from emrisearch.jax_utils import cpu_det_stat, det_stat

        # test signal
        true_values = np.asarray([1e6, 10.0, 0.9, 0.7, 0.1, 1.0])
        deltaT = 5.0
        T_sft = 5e4
        t, hp, hx = create_signal(true_values, T=1.0, deltaT=deltaT)
        samples_per_sft = int(T_sft/deltaT)  # Example number of samples per SFT
        wind = tukey(samples_per_sft, 0.2)  # Example window function
        frequencies = np.fft.rfftfreq(samples_per_sft, deltaT)
        sfts = compute_sfts(hp, deltaT, wind, samples_per_sft)
        snr2 = sft_inner_product(sfts, sfts, frequencies)
        t_alpha = np.arange(sfts.shape[1]) * T_sft  # Example time stamps for SFTs
        true_phi_f_fdot_fddot = get_f_fdot_fddot_back(true_values, t_alpha)
        
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
        
        # EMRI signal
        snr_ref = 30.0  # Desired SNR of the signal
        injection_dict = generate_emri_signal_and_sfts(true_values, 1.0, T_sft, deltaT, snr_ref, 1.0)
        data_sfts = injection_dict['data_sfts']
        signal_sfts = injection_dict['signal_sfts']
        noise_sfts = injection_dict['noise_sfts']
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
        A_alpha = np.where((f_alpha > 5e-4) & (fdot_alpha > 0), 1.0, 0.0)

        A_alpha = jnp.asarray(A_alpha, dtype=jnp.float64)
        f_alpha = jnp.asarray(f_alpha, dtype=jnp.float64)
        fdot_alpha = jnp.asarray(fdot_alpha, dtype=jnp.float64)
        phi_alpha = jnp.asarray(phi_alpha, dtype=jnp.float64)
        data_sfts = jnp.asarray(data_sfts, dtype=jnp.complex128)
        noise_sfts = jnp.asarray(noise_sfts, dtype=jnp.complex128)
        ll_d = cpu_det_stat(data_sfts, A_alpha, phi_alpha, f_alpha, fdot_alpha, T_sft=T_sft)
        ll_s = cpu_det_stat(signal_sfts, A_alpha, phi_alpha, f_alpha, fdot_alpha, T_sft=T_sft)
        ll_n = cpu_det_stat(noise_sfts, A_alpha, phi_alpha, f_alpha, fdot_alpha, T_sft=T_sft)
        ll_d_new = det_stat(data_sfts, f_alpha, fdot_alpha, T_sft=T_sft)
        ll_s_new = det_stat(signal_sfts, f_alpha, fdot_alpha, T_sft=T_sft)
        ll_n_new = det_stat(noise_sfts, f_alpha, fdot_alpha, T_sft=T_sft)
        print(f"Log-likelihood ratio: {ll_d}, {ll_n}, {ll_s}")
        print(f"Log-likelihood ratio (new): {ll_d_new}, {ll_n_new}, {ll_s_new}")
        # Example test: replace with real tests for da_utils
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
