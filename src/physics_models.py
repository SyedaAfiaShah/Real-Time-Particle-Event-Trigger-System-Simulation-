import numpy as np

def generate_random_steps(num_particles: int, dimensions: int, step_size: float) -> np.ndarray:
    """
    Generates Gaussian-distributed random steps for diffusion.
    
    Args:
        num_particles (int): Number of steps to generate.
        dimensions (int): Spatial dimensions.
        step_size (float): The standard deviation of the Gaussian step (D measure).
        
    Returns:
        np.ndarray: Array of shape (num_particles, dimensions) containing the step displacements.
    """
    # Gaussian distribution simulates a random walk (Wiener process)
    return np.random.normal(loc=0.0, scale=step_size, size=(num_particles, dimensions))

def apply_boundary_conditions(positions: np.ndarray, boundary_size: float, boundary_type: str = 'none', is_alive: np.ndarray = None):
    """
    Applies boundary conditions to the given positions in-place.
    
    Args:
        positions (np.ndarray): Particle positions of shape (N, D).
        boundary_size (float): The symmetric size of the boundary [-boundary_size, boundary_size].
        boundary_type (str): 'none', 'reflective', or 'absorbing'.
        is_alive (np.ndarray, optional): Boolean array of alive status. Required for 'absorbing'.
    """
    if boundary_type == 'none' or boundary_size <= 0:
        return
        
    for dim in range(positions.shape[1]):
        if boundary_type == 'reflective':
            # Reflect particles that cross the positive boundary
            overshoot_pos = positions[:, dim] > boundary_size
            positions[overshoot_pos, dim] = 2 * boundary_size - positions[overshoot_pos, dim]
            
            # Reflect particles that cross the negative boundary
            overshoot_neg = positions[:, dim] < -boundary_size
            positions[overshoot_neg, dim] = -2 * boundary_size - positions[overshoot_neg, dim]
            
        elif boundary_type == 'absorbing':
            if is_alive is None:
                raise ValueError("is_alive array must be provided for absorbing boundaries")
            # Particles hitting walls die (are absorbed)
            hit_wall = (positions[:, dim] > boundary_size) | (positions[:, dim] < -boundary_size)
            # The exact position where it hit doesn't perfectly matter if it's dead, 
            # but we clamp it at the boundary to look correct.
            positions[hit_wall & (positions[:, dim] > boundary_size), dim] = boundary_size
            positions[hit_wall & (positions[:, dim] < -boundary_size), dim] = -boundary_size
            
            # Kill the particles that hit the wall
            is_alive[hit_wall] = False

def compute_decay(is_alive: np.ndarray, decay_constant: float, dt: float) -> np.ndarray:
    """
    Determines which alive particles decay in the current time step.
    
    Probability per time step: P(decay) = 1 - exp(-lambda * dt)
    
    Args:
        is_alive (np.ndarray): Boolean array of current alive status.
        decay_constant (float): The decay rate lambda.
        dt (float): Timestep size.
        
    Returns:
        np.ndarray: Boolean array of particles that decayed in THIS timestep.
    """
    if decay_constant <= 0:
        return np.zeros_like(is_alive, dtype=bool)
        
    # Calculate probability of decay in this interval
    p_decay = 1.0 - np.exp(-decay_constant * dt)
    
    # Only active particles can decay
    alive_indices = np.where(is_alive)[0]
    num_alive = len(alive_indices)
    
    if num_alive == 0:
        return np.zeros_like(is_alive, dtype=bool)
        
    # Roll a random number for each alive particle
    rolls = np.random.random(num_alive)
    
    # Determine the indices of those that decay
    decayed_in_dt_indices = alive_indices[rolls < p_decay]
    
    # Create mask of newly decayed
    newly_decayed = np.zeros_like(is_alive, dtype=bool)
    newly_decayed[decayed_in_dt_indices] = True
    
    return newly_decayed
