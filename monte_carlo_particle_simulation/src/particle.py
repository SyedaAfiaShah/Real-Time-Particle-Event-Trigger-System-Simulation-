import numpy as np

class ParticlePopulation:
    """
    A vectorized representation of a population of particles.
    Instead of individual Python objects, we use NumPy arrays for extreme performance.
    """
    
    def __init__(self, num_particles: int, dimensions: int = 2, seed: int = None):
        """
        Initialize the particle population at the origin.
        
        Args:
            num_particles (int): The number of particles to simulate.
            dimensions (int): Dimensionality of the space (e.g., 2 or 3).
            seed (int, optional): Random seed for reproducibility.
        """
        self.num_particles = num_particles
        self.dimensions = dimensions
        
        if seed is not None:
            np.random.seed(seed)
            
        # All particles start at the origin (0, 0, ...)
        self.positions = np.zeros((num_particles, dimensions), dtype=np.float64)
        
        # All particles start alive
        self.is_alive = np.ones(num_particles, dtype=bool)
        
        # Track the time at which each particle decayed (-1 for still alive)
        self.decay_times = np.full(num_particles, -1.0, dtype=np.float64)
        
    def get_alive_indices(self) -> np.ndarray:
        """Returns the indices of currently alive particles."""
        return np.where(self.is_alive)[0]
    
    def get_decayed_indices(self) -> np.ndarray:
        """Returns the indices of decayed particles."""
        return np.where(~self.is_alive)[0]

    def count_alive(self) -> int:
        """Returns the number of currently alive particles."""
        return np.sum(self.is_alive)
        
    def mean_squared_displacement(self) -> float:
        """
        Calculates the mean squared displacement (MSD) of all particles
        (even if bounded or decayed, their final/current position is used).
        
        Returns:
            float: The mean squared displacement from the origin.
        """
        # Distance squared from origin for each particle
        sq_distances = np.sum(self.positions**2, axis=1)
        return np.mean(sq_distances)
