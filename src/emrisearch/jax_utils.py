import jax
import numpy as np
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jax.scipy import special

from interpax import CubicSpline
from functools import partial

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

def psd(freqs):
    pos_freqs = jnp.abs(freqs)
    return jnp.where(pos_freqs < 1e-4, cs(1e-4), cs_eval(pos_freqs))

def dirichlet_kernel(f_0, T_sft):
    """The physically correct kernel for the f_1 = 0 case."""
    return T_sft * jnp.exp(1j * jnp.pi * f_0 * T_sft) * jnp.sinc(f_0 * T_sft)

def fresnel_calculation(f_0, f_1_abs, T_sft):
    """
    The core Fresnel calculation, assuming f_1 is positive.
    This is a helper function to avoid code duplication.
    """
    # This calculation is only valid for f_1_abs > 0
    quot = f_0 / f_1_abs
    factor = jnp.sqrt(2 * f_1_abs)
    
    # Calculate Fresnel integrals at the upper and lower bounds
    Su_lower, Cu_lower = special.fresnel(factor * quot)
    Su_upper, Cu_upper = special.fresnel(factor * (quot + T_sft))
    
    # Combine into the complex result
    complex_fresnel_diff = (Cu_upper - Cu_lower) + 1j * (Su_upper - Su_lower)
    
    # Apply the complex exponential prefactor
    prefactor = jnp.exp(-1j * jnp.pi * f_0 * quot)
    
    return prefactor * complex_fresnel_diff / factor

# Use a decorator to register this as a function with a custom gradient
@partial(jax.custom_vjp, nondiff_argnums=(2,)) # T_sft is not differentiated
def robust_kernel(f_0, f_1, T_sft):
    """
    This is the function we will call. It computes the forward pass.
    """
    abs_f_1 = jnp.abs(f_1)
    return jnp.where(
        abs_f_1 > 0., 
        jnp.where(
            f_1 > 0,
            fresnel_calculation(f_0, abs_f_1, T_sft),
            fresnel_calculation(-f_0, abs_f_1, T_sft).conj(),
        ),
        dirichlet_kernel(f_0, T_sft),
    )

def kernel_fwd(f_0, f_1, T_sft):
    """
    Defines the forward pass for JAX.
    It returns the output and any values needed for the backward pass (residuals).
    """
    output = robust_kernel(f_0, f_1, T_sft)
    return output, (f_0, f_1,)

def kernel_bwd(nondiff_args, residuals, g):
    """
    Defines the backward pass (the VJP).
    `g` is the incoming gradient from the layers above.
    """
    T_sft = nondiff_args
    f_0, f_1 = residuals
    abs_f1 = jnp.abs(f_1)
    
    def dirichlet_for_vjp(f0):
        return dirichlet_kernel(f0, T_sft)
        
    def fresnel_for_vjp(f0, f1_abs):
        return fresnel_calculation(f0, f1_abs, T_sft)
        
    def neg_fresnel_for_vjp(f0, f1_abs):
        return jnp.conj(fresnel_calculation(-f0, f1_abs, T_sft))

    grad_dirichlet_f0, = jax.vjp(dirichlet_for_vjp, f_0)[1](g)
    grad_pos_f0, grad_pos_f1_abs = jax.vjp(fresnel_for_vjp, f_0, abs_f1)[1](g)
    grad_neg_f0, grad_neg_f1_abs = jax.vjp(neg_fresnel_for_vjp, f_0, abs_f1)[1](g)

    # --- Select the correct gradient using the same logic as the forward pass ---
    grad_f0 = jnp.where(
        abs_f1 > 0.,
        jnp.where(f_1 > 0, grad_pos_f0, grad_neg_f0),
        grad_dirichlet_f0
    )
    
    # The gradient of abs(f1) is sign(f1). We apply this via the chain rule.
    grad_f1 = jnp.where(
        abs_f1 > 0.,
        jnp.sign(f_1) * jnp.where(f_1 > 0., grad_pos_f1_abs, grad_neg_f1_abs),
        0.0 # The f1 gradient of the Dirichlet part is 0
    )

    return (grad_f0, grad_f1)

robust_kernel.defvjp(kernel_fwd, kernel_bwd)


# [See Eq. (7) of Tenorio & Gerosa 2025]
def cpu_det_stat(data_sfts, A_alpha, phi_alpha, f_alpha, fdot_alpha, P=100, T_sft=86400., return_terms=False):
    # Non-signal-dependent values are passed here by clousure
    deltaf = 1 / T_sft

    f_k_of_alpha = (f_alpha * T_sft).astype(int)
    k_min_max = f_k_of_alpha + jnp.arange(-P, P + 1)[:, None]

    # Set to 0 whatever gets beyond the range.
    # Note that jax will not complain about out-of-range indexing
    zero_mask = jnp.logical_and(k_min_max >= 0, k_min_max < data_sfts.shape[0]//2)
    # zero_mask = (k_min_max < data_sfts.shape[0]//2)
    kernel = robust_kernel(f_alpha - k_min_max * deltaf, fdot_alpha, T_sft)
    c_alpha = (
        deltaf
        * data_sfts[k_min_max, jnp.arange(data_sfts.shape[1])].conj() / psd(k_min_max * deltaf)
        * kernel
        * zero_mask
    ).sum(axis=0)
    dh_term = 2 * jnp.abs(A_alpha * c_alpha) # jnp.exp(1j * phi_alpha) * 
    hh_term = T_sft * A_alpha**2 / psd(f_alpha) / 2
    # breakpoint()  # For debugging purposes, remove in production
    if return_terms:
        return 0.5 * jnp.nan_to_num(jnp.abs(dh_term)**2 / hh_term).sum(), dh_term, hh_term
    else:
        # If we don't want the terms, just return the statistic
        return 0.5 * jnp.nan_to_num(jnp.abs(dh_term)**2/hh_term).sum()  

def det_stat(data_sfts, f_alpha, fdot_alpha, P=100, T_sft=5e4):
    deltaf = 1 / T_sft
    f_k_continuous = f_alpha * T_sft
    
    # 2. Get the nearest integer bin for the data lookup.
    #    This part is non-differentiable, and that is now intended.
    
    # rounding changese the final value
    # f_k_discrete = jnp.round(f_k_continuous).astype(int)
    # other code does
    f_k_discrete = (f_k_continuous).astype(int)

    # 3. Calculate the DIFFERENTIABLE residual frequency. This is the crucial part.
    #    This carries the gradient from f_alpha.
    f_residual = f_alpha - f_k_discrete * deltaf

    # 4. Create the indices for the lookup using the discrete integer.
    k_min_max = f_k_discrete + jnp.arange(-P, P + 1)[:, None]
    
    # (Good practice) Add a mask to prevent out-of-bounds indexing.
    valid_mask = (k_min_max >= 0) & (k_min_max < data_sfts.shape[0])
    # other code does
    # valid_mask = jnp.logical_or(k_min_max >= 0, k_min_max < data_sfts.shape[0])
    safe_k_min_max = jnp.clip(k_min_max, 0, data_sfts.shape[0] - 1)
    
    # 5. Call the kernel with the DIFFERENTIABLE residual frequency.
    #    The gradient now flows cleanly through f_residual.

    # f_residual removed from the kernel call to match correctly with the original code
    kernel_values = robust_kernel(f_alpha - k_min_max * deltaf, fdot_alpha, T_sft)
    # kernel_values = jnp.nan_to_num(kernels.fresnel_kernel(f_alpha - k_min_max * deltaf, fdot_alpha, T_sft))

    # 6. Perform the lookup and combine terms.
    #    The `where` ensures we don't include contributions from out-of-bounds indices.
    # data_term = data_sfts[safe_k_min_max, jnp.arange(data_sfts.shape[1])].conj()
    # other code does
    data_term = data_sfts[k_min_max, jnp.arange(data_sfts.shape[1])].conj()
    psd_term = psd(k_min_max * deltaf)
    
    c_alpha = (
        deltaf
        * (data_term / psd_term)
        * kernel_values
        * valid_mask
    ).sum(axis=0)

    dh_term_abs = 2 * jnp.abs(c_alpha)
    hh_term = 0.5 * T_sft / psd(f_alpha) # Using continuous f_alpha here is fine
    
    fddot_alpha =  jnp.gradient(fdot_alpha, T_sft)
    delta_phi_approx = jnp.abs(fddot_alpha * T_sft**3 / 6)

    # A_alpha = jnp.where((f_alpha > 5e-4) & (fdot_alpha > 0) & (delta_phi_approx < 0.1), 1.0, 0.0)
    # A_alpha = jnp.where((f_alpha > 10**(-3.5)) & (f_alpha < 10**(-2.0)) & (fdot_alpha > 1e-13) & (fdot_alpha < 1e-6) & (delta_phi_approx < 0.1), 1.0, 0.0)
    A_alpha = jnp.where((f_alpha > 10**(-3.5)) & (fdot_alpha > 1e-13) & (delta_phi_approx < 1.0), 1.0, 0.0)
    # if the sum is non zero, return standard result, otherwise return zero
    return jnp.where(A_alpha.sum()!=0.0, 0.5 * (A_alpha * dh_term_abs**2 / hh_term).sum(), 0.0)


if __name__ == "__main__":
    from emri_utils import create_signal, get_f_fdot_fddot_back
    from search_utils import psd, generate_emri_signal_and_sfts
    from few.utils.constants import YRSID_SI
    from matplotlib import pyplot as plt
    
    T_data = 0.2  # in years
    T_snr = 0.2  # in years
    deltaT = 5.0 # in seconds
    T_sft = 5e4
    Tpl = 0.2 # test shorter
    # Tpl = 0.15 # test longer
    true_values = np.asarray([1e6, 10.0, 0.1, Tpl, 0.1, 1.0])
    t, hp, hx = create_signal(true_values, T=0.1, deltaT=deltaT)
    
    # EMRI signal
    snr_ref = 30.0  # Desired SNR of the signal
    injection_dict = generate_emri_signal_and_sfts(true_values, T_data, T_sft, deltaT, snr_ref, T_snr)
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
    # print("Not perfect agreement due to the delta phi approximation in the fddot_alpha term.")
