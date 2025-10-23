import numpy as np
import math
from scipy.stats import norm
from scipy.integrate import quad

def asymptotic_max_stats(mu, sigma, n):
    """
    Compute approximate E[max] and Var(max) using asymptotic expansion (Gumbel limit) for large n.
    
    Args:
    mu (float): Mean of the Gaussians.
    sigma (float): Standard deviation of the Gaussians.
    n (int): Number of i.i.d. draws.
    
    Returns:
    dict: {'mean': float, 'variance': float}
    """
    logn = math.log(n)
    sqrt2logn = math.sqrt(2 * logn)
    a_n = 1 / sqrt2logn
    loglogn = math.log(logn)
    log4pi = math.log(4 * math.pi)
    correction = (loglogn + log4pi) / (2 * sqrt2logn)
    b_n = sqrt2logn - correction
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    m_n = b_n + gamma * a_n
    pi_sq = math.pi ** 2
    v_n = pi_sq / (12 * logn)
    mean = mu + sigma * m_n
    var = sigma**2 * v_n
    return {'mean': mean, 'variance': var}

def expected_max_integral(mu, sigma, n):
    """
    Compute E[max{X1, ..., Xn}] via numerical quadrature on the standardized integral.
    
    Args:
    mu (float): Mean of the Gaussians.
    sigma (float): Standard deviation of the Gaussians.
    n (int): Number of i.i.d. draws.
    
    Returns:
    float: The expected maximum.
    """
    def integrand(z):
        return n * z * norm.pdf(z) * (norm.cdf(z))**(n - 1)
    
    m_n, _ = quad(integrand, -np.inf, np.inf)
    return mu + sigma * m_n

def var_max_integral(mu, sigma, n):
    """
    Compute Var(max{X1, ..., Xn}) via numerical quadrature.
    Note: mu is not needed for variance, but included for consistency.
    
    Args:
    mu (float): Mean of the Gaussians (unused).
    sigma (float): Standard deviation of the Gaussians.
    n (int): Number of i.i.d. draws.
    
    Returns:
    float: The variance of the maximum.
    """
    def integrand_mean(z):
        return n * z * norm.pdf(z) * (norm.cdf(z))**(n - 1)
    
    m_n, _ = quad(integrand_mean, -np.inf, np.inf)
    
    def integrand_second(z):
        return n * z**2 * norm.pdf(z) * (norm.cdf(z))**(n - 1)
    
    second_moment, _ = quad(integrand_second, -np.inf, np.inf)
    
    var_standard = second_moment - m_n**2
    return sigma**2 * var_standard

def stats_max_monte_carlo(mu, sigma, n, num_trials=100000):
    """
    Estimate E[max] and Var(max) via Monte Carlo simulation.
    
    Args:
    mu (float): Mean of the Gaussians.
    sigma (float): Standard deviation of the Gaussians.
    n (int): Number of i.i.d. draws.
    num_trials (int): Number of simulation trials.
    
    Returns:
    tuple: (mean, variance) from simulation.
    """
    maxes = np.empty(num_trials)
    for i in range(num_trials):
        draws = np.random.normal(mu, sigma, n)
        maxes[i] = np.max(draws)
    return np.mean(maxes), np.var(maxes)

def compute_max_stats(mu, sigma, n, method='integral', num_trials=100000):
    """
    Compute both mean and variance of the maximum using specified method.
    
    Args:
    mu, sigma, n: As above.
    method (str): 'integral' (default, exact numerical), 'monte_carlo' (simulation),
                  or 'asymptotic' (Gumbel approximation for large n).
    num_trials (int): For Monte Carlo only.
    
    Returns:
    dict: {'mean': float, 'variance': float, 'method': str}
    """
    if method == 'asymptotic':
        stats = asymptotic_max_stats(mu, sigma, n)
    elif method == 'monte_carlo':
        mean, var = stats_max_monte_carlo(mu, sigma, n, num_trials)
        stats = {'mean': mean, 'variance': var}
    else:  # 'integral'
        mean = expected_max_integral(mu, sigma, n)
        var = var_max_integral(mu, sigma, n)
        stats = {'mean': mean, 'variance': var}
    
    stats['method'] = method
    return stats

# Example usage
if __name__ == "__main__":
    n = 10000
    mu = 10.0
    sigma = 1.0
    
    # Exact via integrals
    stats_integral = compute_max_stats(mu, sigma, n, method='integral')
    print(f"Integral E[M_{n}]: {stats_integral['mean']:.6f}")
    print(f"Integral Var[M_{n}]: {stats_integral['variance']:.6f}")
    
    # Asymptotic approximation
    stats_asymp = compute_max_stats(mu, sigma, n, method='asymptotic')
    print(f"Asymptotic E[M_{n}]: {stats_asymp['mean']:.6f}")
    print(f"Asymptotic Var[M_{n}]: {stats_asymp['variance']:.6f}")
    
    # Monte Carlo verification
    mc_stats = compute_max_stats(mu, sigma, n, method='monte_carlo')
    print(f"MC E[M_{n}] (10^5 trials): {mc_stats['mean']:.6f}")
    print(f"MC Var[M_{n}] (10^5 trials): {mc_stats['variance']:.6f}")
    
    # Alternative parameters
    alt_stats = compute_max_stats(5.0, 2.0, n, method='integral')
    print(f"Integral E[M_{n}] (mu=5, sigma=2): {alt_stats['mean']:.6f}")
    print(f"Integral Var[M_{n}] (mu=5, sigma=2): {alt_stats['variance']:.6f}")