import numpy as np
import time
from .particle import ParticlePopulation
from .physics_models import generate_random_steps, apply_boundary_conditions, compute_decay

class MonteCarloEngine:
    """
    Core engine for simulating particle transport and decay using Monte Carlo methods.
    """
    
    def __init__(self, 
                 num_particles: int = 1000, 
                 dimensions: int = 2,
                 step_size: float = 1.0,
                 decay_constant: float = 0.05,
                 dt: float = 0.1,
                 boundary_size: float = 0.0,
                 boundary_type: str = 'none',
                 seed: int = None):
        """
        Initializes the simulation engine.
        
        Args:
            num_particles: Number of particles to simulate.
            dimensions: Spatial dimensions.
            step_size: Diffusion standard deviation per unit time. 
                       (Actual step std is scaled by sqrt(dt)).
            decay_constant: Lambda for exponential decay.
            dt: Time step size.
            boundary_size: Half-width of the simulation domain. 0 means infinite.
            boundary_type: 'none', 'reflective', or 'absorbing'.
            seed: Random seed for reproducibility.
        """
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.step_size = step_size
        self.decay_constant = decay_constant
        self.dt = dt
        
        self.boundary_size = boundary_size
        self.boundary_type = boundary_type
        
        # Scaling step size by sqrt(dt) ensures standard diffusion relationship: <x^2> = 2Dt
        self.actual_step_size = self.step_size * np.sqrt(self.dt)
        
        self.population = ParticlePopulation(num_particles, dimensions, seed)
        
        self.current_time = 0.0
        
        # History tracking for analysis
        self.time_history = [0.0]
        self.alive_count_history = [num_particles]
        # To avoid massive memory usage, we might only sample position history periodically
        # But for full trajectory plotting, we keep it. Shape: (num_steps, num_particles, dimensions)
        self.position_history = [self.population.positions.copy()]
        self.msd_history = [0.0]
        
    def step(self):
        """Performs a single simulation step."""
        # 1. Transport (Random Walk)
        # Only move particles that are still alive
        alive_idx = self.population.get_alive_indices()
        
        if len(alive_idx) > 0:
            # Generate steps
            steps = generate_random_steps(len(alive_idx), self.dimensions, self.actual_step_size)
            
            # Update positions
            self.population.positions[alive_idx] += steps
            
        # 2. Apply Boundary Conditions
        if self.boundary_type != 'none':
            apply_boundary_conditions(
                self.population.positions, 
                self.boundary_size, 
                self.boundary_type, 
                self.population.is_alive
            )
            
        # 3. Decay
        newly_decayed = compute_decay(self.population.is_alive, self.decay_constant, self.dt)
        
        # Update states
        self.population.is_alive[newly_decayed] = False
        self.population.decay_times[newly_decayed] = self.current_time + self.dt
        
        # 4. Updates Time and History
        self.current_time += self.dt
        
        self.time_history.append(self.current_time)
        self.alive_count_history.append(self.population.count_alive())
        self.position_history.append(self.population.positions.copy())
        self.msd_history.append(self.population.mean_squared_displacement())

    def run(self, total_time: float) -> dict:
        """
        Runs the simulation for a specified total time.
        
        Args:
            total_time: Total time to simulate.
            
        Returns:
            dict: Simulation results containing history arrays.
        """
        num_steps = int(np.ceil(total_time / self.dt))
        
        start_t = time.time()
        for _ in range(num_steps):
            self.step()
        end_t = time.time()
        
        print(f"Simulation completed {num_steps} steps in {end_t - start_t:.4f} seconds.")
        
        return self.get_results()
        
    def get_results(self) -> dict:
        """Packs history into a results dictionary."""
        return {
            'time': np.array(self.time_history),
            'alive_count': np.array(self.alive_count_history),
            'positions': np.array(self.position_history),
            'msd': np.array(self.msd_history),
            'decay_times': self.population.decay_times.copy()
        }
