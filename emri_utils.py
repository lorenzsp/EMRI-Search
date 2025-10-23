import numpy as np
import few
from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI
from few.trajectory.ode import KerrEccEqFlux
from few.utils.geodesic import get_separatrix
from few.waveform import GenerateEMRIWaveform
from lisatools.detector import EqualArmlengthOrbits
from fastlisaresponse import ResponseWrapper

# Initialize trajectory and waveform generators
traj = EMRIInspiral(func=KerrEccEqFlux)
sum_kwargs = {
    "pad_output": True,
}
waveform = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", sum_kwargs=sum_kwargs)
rhs = KerrEccEqFlux()

def get_response_generator(T, dt, waveform_model="FastKerrEccentricEquatorialFlux"):
    """Function to obtain the waveform generator with LISA response.

    Args:
        T (float): Observation time in years
        dt (float): time step in seconds
        use_gpu (bool): GPU flag
        waveform_model (string): Waveform model to be used

    Returns:
        func: Function that takes in the parameters and returns the waveform
    """

    # define waveform model
    few_gen = GenerateEMRIWaveform(waveform_model, sum_kwargs=dict(pad_output=True), return_list=False)

    use_gpu = few.has_backend('gpu')
    tdi_gen = "2nd generation"
    tdi_kwargs_esa = dict(
        t0=30000.0,
        order=25,
        tdi=tdi_gen,
        tdi_chan="AET",
        orbits=EqualArmlengthOrbits(),
        )

    # with longer signals we care less about this
    t0 = 10000.0  # throw away on both ends when our orbital information is weird
    wave_gen = ResponseWrapper(
        few_gen,
        T,
        dt,
        index_lambda=8,
        index_beta=7,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage='zero',  # removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )
    return wave_gen

def get_f_fdot_fddot_back(params, t_alpha):
    """
    Compute the phase, frequency, and their first and second derivatives for a backward-integrated inspiral trajectory.
    This function integrates the inspiral trajectory backwards in time, starting from the separatrix, and evaluates the orbital phase,
    frequency, and their derivatives at specified time samples. The results are returned for two harmonics (indices 3 and 5 in the
    trajectory output).
    Parameters
    ----------
    params : np.ndarray
        Array of shape (N, 6) or (6,) containing the physical parameters for the inspiral:
        [m1, m2, a, Tpl, e0, x0], where
            m1 : float
                Mass of the primary object.
            m2 : float
                Mass of the secondary object.
            a : float
                Spin parameter.
            Tpl : float
                Plunge time (in years).
            e0 : float
                Initial eccentricity.
            x0 : float
                Initial semi-latus rectum or related parameter.
    t_alpha : np.ndarray
        Array of time samples (in seconds) at which to evaluate the phase and frequency quantities.
    Returns
    -------
    phi : np.ndarray
        Array of shape (2, len(t_alpha)) containing the orbital phases for the two harmonics at each time sample.
    f : np.ndarray
        Array of shape (2, len(t_alpha)) containing the frequencies for the two harmonics at each time sample.
    dotf : np.ndarray
        Array of shape (2, len(t_alpha)) containing the first derivatives of the frequencies (frequency derivatives) for the two harmonics.
    dotdotf : np.ndarray
        Array of shape (2, len(t_alpha)) containing the second derivatives of the frequencies for the two harmonics.
    Notes
    -----
    - The function assumes the existence of global objects and methods such as `traj`, `rhs`, `get_separatrix`, and constants like `YRSID_SI`.
    - Only the time samples within the valid integration range are populated; others remain zero.
    - The function is tailored for EMRI (Extreme Mass Ratio Inspiral) waveform modeling and relies on external trajectory and spline evaluation routines.
    """
    # Define ODE flux and parameters
    m1, m2, a, Tpl, ef, x0 = params.T
    mass_ratio = m1 * m2 / (m1 + m2)**2
    rhs.add_fixed_parameters(params[0], params[1], a)
    # define max time based on t_alpha
    p_0 = traj.inspiral_generator.func.separatrix_buffer_dist + get_separatrix(a, ef, x0) + 1e-6
    # T = t_alpha[-1]/YRSID_SI if Tpl>t_alpha[-1] else Tpl
    backwards_result = traj(m1, m2, a, p_0, ef, x0, T=Tpl, integrate_backwards=True)
    
    t_int = traj.integrator_spline_t
    # time to plunge
    # time_to_plunge = YRSID_SI * T - t_int[::-1]
    
    # new t
    T = np.min([Tpl, t_alpha[-1]/YRSID_SI])
    new_t_alpha = (t_alpha + YRSID_SI * T - t_alpha[-1])[::-1]
    # mask = (new_t_alpha < t_int[-1])&(new_t_alpha > 0.0)
    mask = (new_t_alpha > 0.0)
    # print((new_t_alpha > 0.0), "\n", (new_t_alpha < t_int[-1]))

    shape_alpha = t_alpha.shape[0]
    phi = np.zeros((2, shape_alpha))
    f = np.zeros((2,shape_alpha))
    dotf = np.zeros((2,shape_alpha))
    dotdotf = np.zeros((2,shape_alpha))
    # time
    t_adim = t_int * mass_ratio
    t_alpha_adim = new_t_alpha[mask] * mass_ratio
    
    # cubic spline approach
    # cs_phase = CubicSpline(forwards_result[0], np.stack((forwards_result[4],forwards_result[6])).T)
    # phases
    
    # phi[:,mask] = cs_phase(t_alpha[mask]).T
    phi_ = traj.inspiral_generator.eval_integrator_spline(new_t_alpha[mask])
    phi[0][mask] = phi_[:, 3] + phi_[-1, 3]
    phi[1][mask] = phi_[:, 5] + phi_[-1, 5]
    # phi_ = traj.inspiral_generator.dopr.eval(t_alpha_adim, t_adim, traj.integrator_spline_coeff)
    # phi[0][mask] = -phi_[:, 3] / mass_ratio
    # phi[1][mask] = -phi_[:, 5] / mass_ratio
    
    # frequencies
    # f[:,mask] = cs_phase.derivative(nu=1)(t_alpha[mask]).T/np.pi/2
    f_ = traj.inspiral_generator.eval_integrator_derivative_spline(new_t_alpha[mask], order=1)
    f[0][mask] = -f_[:, 3] / (np.pi * 2)
    f[1][mask] = -f_[:, 5] / (np.pi * 2)
    # f_ = traj.inspiral_generator.dopr.eval_derivative(t_alpha_adim, t_adim, traj.integrator_spline_coeff)
    # f[0][mask] = -f_[:, 3] / (np.pi * 2)
    # f[1][mask] = -f_[:, 5] / (np.pi * 2)

    # derivatives
    # dotf[:,mask] = cs_phase.derivative(nu=2)(t_alpha[mask]).T/np.pi/2
    dotf_ = traj.inspiral_generator.eval_integrator_derivative_spline(new_t_alpha[mask], order=2)
    dotf[0][mask] = dotf_[:, 3] / (np.pi * 2)
    dotf[1][mask] = dotf_[:, 5] / (np.pi * 2)
    # dotf_ = traj.inspiral_generator.dopr.eval_derivative(t_alpha_adim, t_adim, traj.integrator_spline_coeff, order=2) / 2
    # dotf[0][mask] = dotf_[:, 3] / (np.pi * 2) * mass_ratio
    # dotf[1][mask] = dotf_[:, 5] / (np.pi * 2) * mass_ratio

    # second derivatives
    # dotdotf[:,mask] = cs_phase.derivative(nu=3)(t_alpha[mask]).T/np.pi/2
    dotdotf_ = traj.inspiral_generator.eval_integrator_derivative_spline(new_t_alpha[mask], order=3)
    dotdotf[0][mask] = -dotdotf_[:, 3] / (np.pi * 2)
    dotdotf[1][mask] = -dotdotf_[:, 5] / (np.pi * 2)
    # dotdotf_ = traj.inspiral_generator.dopr.eval_derivative(t_alpha_adim, t_adim, traj.integrator_spline_coeff, order=3) / 6
    # dotdotf[0][mask] = -dotdotf_[:, 3] / (np.pi * 2) * mass_ratio * mass_ratio
    # dotdotf[1][mask] = -dotdotf_[:, 5] / (np.pi * 2) * mass_ratio * mass_ratio

    # # check difference
    return phi, f, dotf, dotdotf

def create_signal(params, 
                  dist=1.0,
                  qS=np.pi/3, phiS=0.0, 
                  qK=np.pi/3, phiK=0.0, 
                  Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0,
                  T=1.0, deltaT=10.0, eps=1e-2,
                  return_params=False,
                  randomize=False
                  ):
    """
    Generates a gravitational wave signal for an EMRI (Extreme Mass Ratio Inspiral) system using the provided parameters.

    Parameters
    ----------
    params : array-like
        Array or object containing the EMRI system parameters (m1, m2, a, Tpl, ef, x0).
    dist : float, optional
        Distance to the source (default: 1.0).
    qs : float, optional
        Source sky position polar angle (default: np.pi/3).
    phiS : float, optional
        Source sky position azimuthal angle (default: 0.0).
    qK : float, optional
        Spin orientation polar angle (default: np.pi/3).
    phiK : float, optional
        Spin orientation azimuthal angle (default: 0.0).
    Phi_phi0 : float, optional
        Initial phase in the phi direction (default: 0.0).
    Phi_theta0 : float, optional
        Initial phase in the theta direction (default: 0.0).
    Phi_r0 : float, optional
        Initial phase in the radial direction (default: 0.0).
    T : float, optional
        Total duration of the signal (default: 1.0).
    deltaT : float, optional
        Time step for the waveform (default: 10.0).
    eps : float, optional
        Accuracy parameter for the waveform generation (default: 1e-2).
    return_params : bool, optional
        Whether to return the parameters used for the signal generation (default: False).
    randomize : bool, optional
        Whether to randomize certain parameters (default: False).

    Returns
    -------
    signal : ndarray
        h plus and h cross of the gravitational wave signal.
    """
    
    if randomize:
        qS=np.arccos(np.random.uniform(-1,1))
        phiS=np.random.uniform(0,2*np.pi)
        qK=np.arccos(np.random.uniform(-1,1))
        phiK=np.random.uniform(0,2*np.pi)
        Phi_phi0=np.random.uniform(0,2*np.pi)
        Phi_theta0=np.random.uniform(0,2*np.pi)
        Phi_r0=np.random.uniform(0,2*np.pi)

    m1, m2, a, Tpl, ef, x0 = params.T
    p_f = traj.inspiral_generator.func.separatrix_buffer_dist + get_separatrix(a, ef, 1.0) + 1e-6
    backwards_result = traj(m1, m2, a, p_f, ef, x0, T=Tpl, integrate_backwards=True)
    p0_true = backwards_result[1][-1]
    e0_true = backwards_result[2][-1]
    x0_true = backwards_result[3][-1]
    try:
        waveform = get_response_generator(T, deltaT)
        print("Using response generator")
        wave_out = waveform(m1, m2, a, p0_true, e0_true, x0_true, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=deltaT, padoutput=True, eps=eps)
        wave_out = wave_out[0] + 1j * wave_out[1]
    except:
        print("Using hp hx generator")
        wave_out = waveform(m1, m2, a, p0_true, e0_true, x0_true, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=deltaT, padoutput=True, eps=eps)
    
    if few.has_backend('gpu'):
        signal = wave_out.get()
        print("Waveform generated on GPU")
    else:
        signal = wave_out.copy()
        print("Waveform generated on CPU")

    time_vector = np.arange(signal.shape[0]) * deltaT
    params_dict = {'m1': m1, 'm2': m2,'a': a, 'Tpl': Tpl, 'ef': ef, 'e0': e0_true, 'pf': p_f, 'p0': p0_true, "dist": dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK, 'Phi_phi0': Phi_phi0, 'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0}
    if return_params:
        return time_vector, signal.real, signal.imag, params_dict
    else:
        return time_vector, signal.real, signal.imag

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    Tpl = 0.5 # 1.5 test different time to plunges
    true_values = np.asarray([1e6, 10.0, 0.9, Tpl, 0.2, 1.0])
    t_alpha = np.arange(0, YRSID_SI, 86400)
    phi, f, dotf, dotdotf = get_f_fdot_fddot_back(true_values, t_alpha)
    t, hp, hx = create_signal(true_values, T=0.01)
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(t, hp, label='h_plus')
    # plt.plot(t, hx, label='h_cross')
    # plt.xlabel('Time (s)')
    # plt.legend()
    # plt.show()

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    labels = [r'$\Phi$', r'$f$', r'$\dot{f}$', r'$\ddot{f}$']
    y_datas = [phi, f, dotf, dotdotf]

    for i, ax in enumerate(axs):
        ax.axvline(true_values[3]*YRSID_SI, color='black', label='Plunge')
        ax.plot(t_alpha, y_datas[i][0], label=r'$\Phi$')
        ax.plot(t_alpha, y_datas[i][1], label=r'$r$', linestyle='--')
        ax.set_ylabel(labels[i])
        ax.legend()
        ax.grid(True)
        

    axs[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()