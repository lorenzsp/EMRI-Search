import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


def to_unit_cube(x, lower, upper):
    """Map variables x from [lower, upper] to [0, 1]."""
    return (x - lower) / (upper - lower)

def from_unit_cube(u, lower, upper):
    """Map variables u from [0, 1] to [lower, upper]."""
    return lower + u * (upper - lower)

def draw_sobol_distribution(dimensions=2, n_samples=1024, seed=None):
    """
    Draws samples from a Sobol distribution and plots them.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the Sobol sequence.
    n_samples : int
        Number of samples to generate.
    seed : int or None
        Random seed for reproducibility.
    """
    sampler = qmc.Sobol(d=dimensions, rng=np.random.default_rng(seed))
    samples = sampler.random(n=n_samples)
    return samples

class ParameterScaler:
    def __init__(self, bounds, seed=2601):
        """
        Initialize the scaler with parameter bounds.

        Parameters
        ----------
        bounds : list of tuple
            List of (min, max) tuples for each parameter.
        """
        self.bounds = np.array(bounds)
        self.mins = self.bounds[:, 0]
        self.maxs = self.bounds[:, 1]
        self.ranges = self.maxs - self.mins
        self.seed = seed

    def to_unit(self, params):
        """
        Convert parameters from their original space to [0, 1] interval.

        Parameters
        ----------
        params : array-like
            Parameters in the original space.

        Returns
        -------
        np.ndarray
            Parameters scaled to [0, 1].
        """
        params = np.asarray(params)
        return (params - self.mins) / self.ranges

    def from_unit(self, unit_params):
        """
        Convert parameters from [0, 1] interval back to original space.

        Parameters
        ----------
        unit_params : array-like
            Parameters in [0, 1] interval.

        Returns
        -------
        np.ndarray
            Parameters in the original space.
        """
        unit_params = np.asarray(unit_params)
        return unit_params * self.ranges + self.mins

    def draw_samples(self, n_samples):
        """
        Draw samples from the parameter space.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Samples in the original parameter space.
        """
        unit_samples = draw_sobol_distribution(dimensions=self.bounds.shape[0], n_samples=n_samples, seed=self.seed)
        return self.from_unit(unit_samples)

if __name__ == "__main__":
    bounds = np.asarray([(3e5, 3e6), (1., 150.), (0.0, 0.998), (0.1, 1.0), (0.0, 0.2)])
    ps = ParameterScaler(bounds)
    n_samples = 1000
    dimensions = bounds.shape[0]
    params = ps.draw_samples(n_samples=n_samples)
    
    uniform_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, dimensions))
    # plot first two parameters
    plt.figure(figsize=(8, 6))
    plt.scatter(params[:, 0], params[:, 1], alpha=0.5, label='Sobol Samples', color='blue')
    plt.scatter(uniform_samples[:, 0], uniform_samples[:, 1], alpha=0.5, label='Uniform Samples', color='red')
    plt.legend()
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.xlim(bounds[0, 0], bounds[0, 1])
    plt.ylim(bounds[1, 0], bounds[1, 1])
    plt.grid()
    plt.tight_layout()
    plt.show()
