import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import src even if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import MonteCarloEngine
from src.utils import get_summary_statistics

def run_parameter_sweep(lambdas: list, output_dir: str = "results"):
    """
    Runs a parameter sweep over decay constants (lambda) and plots the final survival curve
    and half-lives.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Running parameter sweep over decay constants: {lambdas}")
    
    half_lives = []
    survival_rates = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Static parameters
    total_time = 15.0
    dt = 0.1
    num_particles = 2000
    
    for i, lam in enumerate(lambdas):
        print(f"Simulating lambda = {lam:.2f} ...")
        engine = MonteCarloEngine(
            num_particles=num_particles,
            decay_constant=lam,
            dt=dt,
            seed=42 + i
        )
        
        results = engine.run(total_time)
        stats = get_summary_statistics(results)
        
        half_lives.append(stats['empirical_half_life'])
        survival_rates.append(stats['final_alive_percentage'])
        
        # Plot this lambda's survival curve
        time_arr = results['time']
        alive_count = results['alive_count'] / num_particles * 100
        ax.plot(time_arr, alive_count, label=f"$\\lambda$ = {lam:.2f}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Survival Percentage (%)")
    ax.set_title("Survival Curve Sweep (Varying $\\lambda$)")
    ax.legend(title="Decay Constant")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    filepath_curves = os.path.join(output_dir, "sweep_survival_curves.png")
    fig.tight_layout()
    fig.savefig(filepath_curves, dpi=150)
    plt.close(fig)
    print(f"Saved survival curves to {filepath_curves}")
    
    # Plot half life vs lambda
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(lambdas, half_lives, 'o-', color='purple', linewidth=2)
    # Theory: t_1/2 = ln(2) / lambda
    valid_lams = np.array([l for l in lambdas if l > 0])
    ax2.plot(valid_lams, np.log(2) / valid_lams, 'k--', label="Theoretical $\\ln(2)/\\lambda$")
    
    ax2.set_xlabel("Decay Constant ($\\lambda$)")
    ax2.set_ylabel("Empirical Half Life (s)")
    ax2.set_title("Half Life vs Decay Constant")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    filepath_hl = os.path.join(output_dir, "sweep_half_lives.png")
    fig2.tight_layout()
    fig2.savefig(filepath_hl, dpi=150)
    plt.close(fig2)
    print(f"Saved half lives to {filepath_hl}")

if __name__ == "__main__":
    test_lambdas = [0.05, 0.1, 0.2, 0.5, 0.8]
    run_parameter_sweep(test_lambdas, output_dir="../results")
    print("Parameter sweep complete.")
