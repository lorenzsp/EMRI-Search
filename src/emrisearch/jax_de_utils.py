import jax
import jax.numpy as jnp

@jax.jit
def selection(fitness, population, trial_fitness, trial_population):
    """
    Selects individuals for the next generation based on fitness improvement.

    Compares the fitness of trial individuals to the current population and selects the better individuals.
    If a trial individual's fitness is better (lower), it replaces the corresponding individual in the population.

    Args:
        fitness (jnp.ndarray): Fitness values of the current population. Shape: (population_size,)
        population (jnp.ndarray): Current population. Shape: (population_size, num_features)
        trial_fitness (jnp.ndarray): Fitness values of the trial population. Shape: (population_size,)
        trial_population (jnp.ndarray): Trial population. Shape: (population_size, num_features)

    Returns:
        selected_population (jnp.ndarray): Updated population after selection. Shape: (population_size, num_features)
        selected_fitness (jnp.ndarray): Updated fitness values after selection. Shape: (population_size,)
        improved (jnp.ndarray): Boolean array indicating which individuals were improved. Shape: (population_size,)
    """
    improved = trial_fitness < fitness
    selected_population = jnp.where(improved[:, None], trial_population, population)
    selected_fitness = jnp.where(improved, trial_fitness, fitness)
    return selected_population, selected_fitness, improved

@jax.jit
def differential_evolution_step(key, population, fitness=None, 
                               elitism=True, crossover_rate=0.9, 
                               differential_weight=0.8, num_diff=1):
    """
    Perform one step of differential evolution on a population.
    
    Args:
        key: JAX random key
        population: Array of shape (population_size, num_dims)
        fitness: Array of shape (population_size,) - optional, computed if None
        elitism: Boolean, whether to use best member as base vector
        crossover_rate: Float in [0, 1], crossover probability
        differential_weight: Float in [0, 2], differential weight factor
        num_diff: Int, number of difference vectors to use
    
    Returns:
        new_population: Evolved population of same shape as input
    """
    population_size, num_dims = population.shape
    
    # Find best member if using elitism
    if fitness is not None:
        best_index = jnp.argmin(fitness)
    else:
        best_index = 0  # Default to first member if no fitness provided
    
    keys = jax.random.split(key, population_size)
    member_ids = jnp.arange(population_size)

    def _evolve_member(key, member_id):
        """Evolve a single member of the population."""
        x = population[member_id]
        
        key_a, key_R, key_r, key_ab = jax.random.split(key, 4)
        
        # Create probability vector excluding current member
        p = jnp.ones(population_size).at[member_id].set(0.0)
        
        # Select base vector
        a_index = jax.random.choice(key_a, population_size, p=p)
        
        # Use elitism: base vector is best member or random
        a_index = jnp.where(elitism, best_index, a_index)
        a = population[a_index]
        
        # Crossover mask: ensure at least one dimension is mutated
        R = jax.random.choice(key_R, num_dims)
        R = jax.nn.one_hot(R, num_dims)
        r = jax.random.uniform(key_r, (num_dims,))
        mask = jnp.logical_or(r < crossover_rate, R)
        
        # Apply differential mutation
        mutant = a
        p_diff = p.at[a_index].set(0.0)  # Exclude base vector from difference vectors
        
        for _ in range(num_diff):
            key_ab, subkey = jax.random.split(key_ab)
            # Select two distinct random members
            indices = jax.random.choice(subkey, population_size, (2,), replace=False, p=p_diff)
            b, c = population[indices[0]], population[indices[1]]
            
            # Update mutant vector
            mutant = mutant + differential_weight * (b - c)
            
            # Update probability to exclude selected members
            p_diff = p_diff.at[indices].set(0.0)
        
        # Apply crossover: use mutant where mask is True, original where False
        # trial = jnp.where(mask, mutant, x)
        return mutant

    # Evolve all members in parallel
    new_population = jax.vmap(_evolve_member)(keys, member_ids)
    return new_population

def differential_evolution_with_selection(key, population, fitness, objective_fn,
                                        bounds=None, elitism=True, crossover_rate=0.9,
                                        differential_weight=0.8, num_diff=1):
    """
    Complete DE step including selection (survival of the fittest) with optional bounds.
    
    Args:
        key: JAX random key
        population: Current population
        fitness: Current fitness values
        objective_fn: Function to evaluate fitness of new candidates
        bounds: Array of shape (num_dims, 2) with [lower, upper] bounds (optional)
        elitism, crossover_rate, differential_weight, num_diff: DE parameters
    
    Returns:
        new_population: Updated population after selection
        new_fitness: Updated fitness values
    """
    # Generate trial population
    trial_population = differential_evolution_step(
        key, population, fitness, elitism, crossover_rate, 
        differential_weight, num_diff
    )
    
    # Apply bounds if provided
    if bounds is not None:
        lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
        trial_population = jnp.clip(trial_population, lower_bounds, upper_bounds)
    
    # Evaluate trial population
    trial_fitness = jax.vmap(objective_fn)(trial_population)

    new_population, new_fitness, improved = selection(fitness, population, trial_fitness, trial_population)
    return new_population, new_fitness, improved
