import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(results: dict, num_to_plot: int = 10, dimensions: int = 2):
    """
    Plots the trajectories of a subset of particles.
    
    Args:
        results (dict): Output from MonteCarloEngine.run().
        num_to_plot (int): Number of particle trajectories to visualize.
        dimensions (int): Either 2 or 3.
        
    Returns:
        matplotlib.figure.Figure
    """
    positions = results['positions']
    total_particles = positions.shape[1]
    
    plot_idx = np.random.choice(total_particles, size=min(num_to_plot, total_particles), replace=False)
    
    fig = plt.figure(figsize=(10, 8))
    
    if dimensions == 2:
        ax = fig.add_subplot(111)
        for i in plot_idx:
            # Mask out decayed parts - we drop points where positions are stationary after decay?
            # Actually, our simulation stops updating them when decay happens.
            ax.plot(positions[:, i, 0], positions[:, i, 1], alpha=0.6, linewidth=1.5)
            # Mark start
            ax.scatter(positions[0, i, 0], positions[0, i, 1], color='green', s=20, zorder=5)
            # Mark end
            ax.scatter(positions[-1, i, 0], positions[-1, i, 1], color='red', s=20, zorder=5)
            
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Particle Trajectories (2D Random Walk)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')
        
    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in plot_idx:
            ax.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2], alpha=0.6, linewidth=1.5)
            
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("Particle Trajectories (3D Random Walk)")
        
    fig.tight_layout()
    return fig

def plot_survival_curve(results: dict, decay_constant: float):
    """
    Plots the survival curve (alive particles vs time) and compares to theory.
    """
    time_arr = results['time']
    alive_count = results['alive_count']
    N0 = alive_count[0]
    
    # Theoretical curve: N(t) = N0 * exp(-lambda * t)
    theory_count = N0 * np.exp(-decay_constant * time_arr)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_arr, alive_count, 'b-', label='Simulation', linewidth=2)
    ax.plot(time_arr, theory_count, 'r--', label='Theoretical $N_0 e^{-\\lambda t}$', linewidth=2)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Alive Particles")
    ax.set_title("Particle Survival Curve")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    # Log scale is often better for exponential decay
    ax.set_yscale('log')
    
    fig.tight_layout()
    return fig

def plot_msd(results: dict, step_size: float, dimensions: int):
    """
    Plots the Mean Squared Displacement (MSD) vs Time.
    """
    time_arr = results['time']
    msd = results['msd']
    
    # Theory: MSD = 2 * dimensions * D * t
    # where D is the diffusion coefficient.
    # We generated steps with N(0, step_size * sqrt(dt)). 
    # Variance of step is step_size^2 * dt.
    # Variance of position after n steps (time t): n * step_size^2 * dt = step_size^2 * t.
    # Also, for 1D diffusion, variance = 2Dt. So D = step_size^2 / 2.
    # In 'dimensions', theory MSD = dimensions * step_size^2 * t
    theory_msd = dimensions * (step_size**2) * time_arr
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_arr, msd, 'g-', label='Simulation MSD', linewidth=2)
    ax.plot(time_arr, theory_msd, 'k--', label='Theoretical MSD ($2nDt$)', linewidth=2)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Squared Displacement")
    ax.set_title("Diffusion Analysis (Mean Squared Displacement)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    return fig
    
def get_summary_statistics(results: dict):
    """Calculates summary statistics from a simulation run."""
    decayed_particles = results['decay_times'][results['decay_times'] > 0]
    half_life = np.median(decayed_particles) if len(decayed_particles) > 0 else np.inf
    
    return {
        "final_alive_percentage": results['alive_count'][-1] / results['alive_count'][0] * 100,
        "empirical_half_life": half_life,
        "final_msd": results['msd'][-1]
    }
