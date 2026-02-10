import numpy as np
from eryn.moves.mh import MHMove
import torch

class DEMove(MHMove):
    """A Metropolis step with a Differential Evolution proposal function.

    Args:
        chain_dict (dict): Dictionary with branch names as keys and chain arrays as values.
            Each chain should have shape (n_samples, n_params) for that branch.
        F (float, optional): The differential weight. (default: 0.5)
        CR (float, optional): The crossover probability. (default: 0.9)
        use_current_state (bool, optional): Whether to use current state in DE formula. (default: True)
        crossover (bool, optional): Whether to perform crossover. (default: False)
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)
    """

    def __init__(self, chain_dict=None, F=0.5, CR=0.9, use_current_state=True, crossover=False, **kwargs):
        self.chain_dict = chain_dict or {}
        self.F = F
        self.CR = CR
        self.use_current_state = use_current_state
        self.crossover = crossover
        
        # Create DE proposals for each branch
        self.all_proposal = {}
        for name in self.chain_dict.keys():
            self.all_proposal[name] = _de_proposal(
                self.chain_dict[name], F, CR, use_current_state, crossover
            )
        
        super(DEMove, self).__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from Differential Evolution

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max] representing which
                leaves are currently being used. (default: ``None``)
            **kwargs (ignored): This is added for compatibility. It is ignored in this function.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, np.ndarray)
        """

        # initialize output
        q = {}
        for name, coords in branches_coords.items():
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # get the proposal for this branch
            if name in self.all_proposal:
                proposal_fn = self.all_proposal[name]
            else:
                # Fall back to using current state if no chain provided
                proposal_fn = _de_proposal(None, self.F, self.CR, self.use_current_state, self.crossover)
            
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords, _ = proposal_fn(coords[inds_here], random)

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            q = self.periodic.wrap(
                {
                    name: tmp.reshape((ntemps * nwalkers,) + tmp.shape[-2:])
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )

            q = {
                name: tmp.reshape(
                    (
                        ntemps,
                        nwalkers,
                    )
                    + tmp.shape[-2:]
                )
                for name, tmp in q.items()
            }

        return q, np.zeros((ntemps, nwalkers))


class _de_proposal(object):
    """Differential Evolution proposal class"""
    
    def __init__(self, chain, F, CR, use_current_state, crossover):
        self.chain = chain
        self.F = F
        self.CR = CR
        self.use_current_state = use_current_state
        self.crossover = crossover

    def get_factor(self, rng):
        """Get random scaling factor"""
        prob = rng.random()
        if prob > 0.5:
            # Random factor
            return rng.uniform(0.1, 2.0)
        else:
            # Gaussian factor
            return np.abs(rng.normal(0, 1.68/np.sqrt(5)))  # Assuming 5D parameter space

    def get_CR(self, rng):
        """Get crossover probability"""
        prob = rng.random()
        if prob > 0.5:
            return rng.uniform(0.5, 1.0)
        else:
            return self.CR

    def __call__(self, x0, rng):
        """Generate DE proposal
        
        Args:
            x0: Current state array with shape (n_walkers, n_params)
            rng: Random number generator
            
        Returns:
            tuple: (proposed_state, log_factors)
        """
        n_walkers, n_params = x0.shape
        
        # Get dynamic parameters
        # F = self.get_factor(rng)
        # CR = self.get_CR(rng)
        F = self.F
        CR = self.CR
        
        # Use provided chain or current state
        chain_to_use = self.chain if self.chain is not None else x0.copy()
        
        # Call the DE proposal function
        proposed_state = propose_DE(
            x0, chain_to_use, F=F, CR=CR, 
            use_current_state=self.use_current_state, 
            crossover=self.crossover
        )
        
        return proposed_state, np.zeros(n_walkers)


def propose_DE(current_state, chain, F=0.5, CR=0.9, use_current_state=True, crossover=False):
    """
    Provides a proposal for MCMC using Differential Evolution (DE/rand/1).

    Parameters:
        current_state (numpy.ndarray): The current state of the MCMC chain. Shape: (n_walkers, n_params).
        chain (numpy.ndarray): The chain from which to take the mutant. Shape: (n_mutants, n_params).
        F (float): The differential weight (default is 0.5), in [0,2].
        CR (float): The crossover probability (default is 0.9), in [0,1].

    Returns:
        numpy.ndarray: The proposed state. Shape: (n_walkers, n_params).
    """
    n_walkers, n_params = current_state.shape

    # Randomly select three distinct indices for each walker
    indices = np.random.choice(chain.shape[0], size=(n_walkers, 3), replace=True)
    
    # Generate mutant vectors using DE/rand/1
    if use_current_state:
        mutant_vectors = current_state + F * (chain[indices[:, 1]] - chain[indices[:, 2]])
    else:
        mutant_vectors = chain[indices[:, 0]] + F * (chain[indices[:, 1]] - chain[indices[:, 2]])

    # Perform crossover with the current state to create the proposed state
    if crossover:
        crossover_mask = (np.random.rand(n_walkers, n_params) <= CR) | (np.arange(n_params) == np.random.randint(n_params, size=(n_walkers, 1)))
    else:
        # to update all
        crossover_mask = np.ones((n_walkers, n_params), dtype=bool)
    proposed_state = np.where(crossover_mask, mutant_vectors, current_state)
    
    return proposed_state

from copy import deepcopy

class SBIDistribution(object):
    """Generate a distribution based on an SBI posterior

    Args:
        sbi_posterior: The SBI posterior object with sample() and log_prob() methods

    Raises:
        ValueError: If ``sbi_posterior`` doesn't have required methods.
    """

    def __init__(self, sbi_posterior, min_log_prob=0.0):
        if not hasattr(sbi_posterior, 'sample') or not hasattr(sbi_posterior, 'log_prob'):
            raise ValueError("sbi_posterior must have 'sample' and 'log_prob' methods.")

        self.sbi_posterior = sbi_posterior
        self.min_log_prob = min_log_prob

    def rvs(self, size=1):
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")

        if isinstance(size, int):
            size = (size,)

        # Sample from SBI posterior
        samples = self.sbi_posterior.sample(size).numpy()
        
        return samples

    def pdf(self, x):
        """Compute probability density using exp(log_prob)"""
        
        # Convert to tensor if needed
        if isinstance(x, (np.ndarray, list)):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x
            
        # Get log probability and convert to probability
        log_prob = self.sbi_posterior.log_prob(x_tensor)
        prob = torch.exp(log_prob)
        
        # Convert back to numpy array
        return prob.numpy()

    def logpdf(self, x):
        """Compute log probability density using SBI posterior log_prob"""
        
        # Convert to tensor if needed
        if isinstance(x, (np.ndarray, list)):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x
            
        # Get log probability from SBI posterior
        log_prob = self.sbi_posterior.log_prob(x_tensor)
        
        # where it is smaller than 1 set to -inf, where bigger set to 0
        if self.min_log_prob is not None:
            log_prob = torch.where(log_prob < self.min_log_prob, torch.tensor(-np.inf), torch.tensor(0.0))

        # Convert back to numpy array
        return log_prob.numpy()

    def copy(self):
        return deepcopy(self)

def sbi_dist(sbi_posterior, min_log_prob=0.0):
    """Generate a distribution based on an SBI posterior

    Args:
        sbi_posterior: The SBI posterior object with sample() and log_prob() methods

    Returns:
        :class:`SBIDistribution`: SBI-based distribution.

    """
    dist = SBIDistribution(sbi_posterior, min_log_prob=min_log_prob)

    return dist


class SBIMove(MHMove):
    """A Metropolis step with an SBI posterior proposal function.

    Args:
        sbi_dict (dict): Dictionary with branch names as keys and SBI posterior objects as values.
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)
    """

    def __init__(self, sbi_dict, **kwargs):
        self.sbi_dict = sbi_dict
        
        # Create SBI proposals for each branch
        self.all_proposal = {}
        for name, sbi_posterior in self.sbi_dict.items():
            self.all_proposal[name] = _sbi_proposal(sbi_posterior)
        
        super(SBIMove, self).__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from SBI posterior

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max] representing which
                leaves are currently being used. (default: ``None``)
            **kwargs (ignored): This is added for compatibility. It is ignored in this function.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, np.ndarray)
        """

        # initialize output
        q = {}
        for name, coords in branches_coords.items():
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # get the proposal for this branch
            if name in self.all_proposal:
                proposal_fn = self.all_proposal[name]
            else:
                raise ValueError(f"No SBI posterior provided for branch {name}")
            
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords, factors = proposal_fn(coords[inds_here], random)

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            q = self.periodic.wrap(
                {
                    name: tmp.reshape((ntemps * nwalkers,) + tmp.shape[-2:])
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )

            q = {
                name: tmp.reshape(
                    (
                        ntemps,
                        nwalkers,
                    )
                    + tmp.shape[-2:]
                )
                for name, tmp in q.items()
            }

        return q, factors.reshape((ntemps, nwalkers))


class _sbi_proposal(object):
    """SBI posterior proposal class"""
    
    def __init__(self, sbi_posterior):
        self.sbi_posterior = sbi_posterior

    def __call__(self, x0, rng):
        """Generate SBI proposal
        
        Args:
            x0: Current state array with shape (n_walkers, n_params)
            rng: Random number generator
            
        Returns:
            tuple: (proposed_state, log_factors)
        """
        n_walkers, n_params = x0.shape
        
        # Sample new proposals directly from SBI posterior
        proposed_state = self.sbi_posterior.sample((n_walkers,),show_progress_bars=False)
        factors = self.sbi_posterior.log_prob(proposed_state).numpy() - self.sbi_posterior.log_prob(x0).numpy()
        return proposed_state.numpy(), factors


class SBIMixtureMove(MHMove):
    """A Metropolis step that mixes SBI proposals with local moves.

    Args:
        sbi_dict (dict): Dictionary with branch names as keys and SBI posterior objects as values.
        sbi_prob (float, optional): Probability of using SBI proposal vs local move. (default: 0.5)
        local_scale (float, optional): Scale for local Gaussian proposals. (default: 0.1)
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)
    """

    def __init__(self, sbi_dict, sbi_prob=0.5, local_scale=0.1, **kwargs):
        self.sbi_dict = sbi_dict
        self.sbi_prob = sbi_prob
        self.local_scale = local_scale
        
        # Create SBI proposals for each branch
        self.all_proposal = {}
        for name, sbi_posterior in self.sbi_dict.items():
            self.all_proposal[name] = _sbi_mixture_proposal(
                sbi_posterior, sbi_prob, local_scale
            )
        
        super(SBIMixtureMove, self).__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from SBI posterior mixture

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max] representing which
                leaves are currently being used. (default: ``None``)
            **kwargs (ignored): This is added for compatibility. It is ignored in this function.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, np.ndarray)
        """

        # initialize output
        q = {}
        for name, coords in branches_coords.items():
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # get the proposal for this branch
            if name in self.all_proposal:
                proposal_fn = self.all_proposal[name]
            else:
                raise ValueError(f"No SBI posterior provided for branch {name}")
            
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords, _ = proposal_fn(coords[inds_here], random)

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            q = self.periodic.wrap(
                {
                    name: tmp.reshape((ntemps * nwalkers,) + tmp.shape[-2:])
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )

            q = {
                name: tmp.reshape(
                    (
                        ntemps,
                        nwalkers,
                    )
                    + tmp.shape[-2:]
                )
                for name, tmp in q.items()
            }

        return q, np.zeros((ntemps, nwalkers))


class _sbi_mixture_proposal(object):
    """SBI posterior mixture proposal class"""
    
    def __init__(self, sbi_posterior, sbi_prob, local_scale):
        self.sbi_posterior = sbi_posterior
        self.sbi_prob = sbi_prob
        self.local_scale = local_scale

    def __call__(self, x0, rng):
        """Generate SBI mixture proposal
        
        Args:
            x0: Current state array with shape (n_walkers, n_params)
            rng: Random number generator
            
        Returns:
            tuple: (proposed_state, log_factors)
        """
        n_walkers, n_params = x0.shape
        
        # Decide which walkers get SBI proposals vs local moves
        use_sbi = rng.random(n_walkers) < self.sbi_prob
        
        # Initialize with current state
        proposed_state = x0.copy()
        
        # Apply SBI proposals to selected walkers
        if np.any(use_sbi):
            n_sbi = np.sum(use_sbi)
            sbi_proposals = self.sbi_posterior.sample((n_sbi,)).numpy()
            proposed_state[use_sbi] = sbi_proposals
        
        # Apply local Gaussian moves to remaining walkers
        if np.any(~use_sbi):
            local_proposals = x0[~use_sbi] + self.local_scale * rng.randn(np.sum(~use_sbi), n_params)
            proposed_state[~use_sbi] = local_proposals
        
        return proposed_state, np.zeros(n_walkers)