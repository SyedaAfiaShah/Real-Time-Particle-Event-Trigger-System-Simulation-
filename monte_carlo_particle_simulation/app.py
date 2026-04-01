import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Try to import our simulation engine
try:
    from src.simulation import MonteCarloEngine
    from src.utils import plot_trajectories, plot_survival_curve, plot_msd, get_summary_statistics
except ModuleNotFoundError:
    st.error("Could not import src modules. Ensure app.py is in the monte_carlo_particle_simulation directory and executed from there.")
    st.stop()

st.set_page_config(page_title="Monte Carlo Particle Simulation", layout="wide")

st.title("Monte Carlo Simulation of Particle Transport and Decay")
st.markdown("""
This application demonstrates a **Monte Carlo framework** for tracking particle diffusion and exponential decay. 
It utilizes a high-performance **vectorized NumPy engine** to simulate large populations of particles.
""")

st.sidebar.header("Simulation Parameters")

with st.sidebar.form("sim_params"):
    num_particles = st.number_input("Number of Particles", min_value=10, max_value=20000, value=2000, step=100)
    dimensions = st.selectbox("Spatial Dimensions", options=[2, 3], index=0)
    step_size = st.slider("Diffusion Step Size / Standard Dev", min_value=0.1, max_value=5.0, value=1.0)
    decay_constant = st.slider("Decay Constant (lambda)", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
    
    st.markdown("---")
    
    st.subheader("Time settings")
    dt = st.number_input("Time Step (dt)", min_value=0.01, max_value=1.0, value=0.1)
    total_time = st.number_input("Total Simulation Time", min_value=1.0, max_value=100.0, value=10.0)
    
    st.markdown("---")
    
    st.subheader("Environment")
    boundary_type = st.selectbox("Boundary Condition", options=['none', 'reflective', 'absorbing'], index=0)
    boundary_size = st.number_input("Boundary Size (Half-width)", min_value=0.0, max_value=100.0, value=10.0)
    
    submit = st.form_submit_button("Run Simulation")

if submit:
    with st.spinner(f"Simulating {num_particles} particles over {total_time} seconds (dt={dt})..."):
        # Initialize engine
        engine = MonteCarloEngine(
            num_particles=num_particles,
            dimensions=dimensions,
            step_size=step_size,
            decay_constant=decay_constant,
            dt=dt,
            boundary_size=boundary_size,
            boundary_type=boundary_type,
            seed=42 # fixed seed for repeatable demo
        )
        
        # Run simulation
        results = engine.run(total_time)
        
    st.success("Simulation Complete!")
    
    # Calculate statistics
    stats = get_summary_statistics(results)
    
    st.header("Key Simulation Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Survival Rate", f"{stats['final_alive_percentage']:.2f}%")
    col2.metric("Empirical Half-Life", f"{stats['empirical_half_life']:.2f}s" if stats['empirical_half_life'] != np.inf else "N/A")
    col3.metric("Final MSD", f"{stats['final_msd']:.2f}")

    # Tabs for visualization
    tab1, tab2, tab3, tab4 = st.tabs(["Particle Trajectories", "Survival Curve", "Mean Squared Displacement", "Parameter Sweep (λ)"])
    
    with tab1:
        st.subheader("Random Walk Trajectories")
        st.markdown("Displays up to 10 particle paths. Starts are marked in green, endpoints in red.")
        fig_traj = plot_trajectories(results, num_to_plot=10, dimensions=dimensions)
        st.pyplot(fig_traj)
        
    with tab2:
        st.subheader("Particle Decay Law")
        st.markdown(r"Compares simulated survival against $N_0 e^{-\lambda t}$.")
        fig_decay = plot_survival_curve(results, decay_constant)
        st.pyplot(fig_decay)
        
    with tab3:
        st.subheader("Diffusion Analysis (MSD)")
        st.markdown("Mean Squared Displacement vs time. For unbounded random walks, this should be linear.")
        fig_msd = plot_msd(results, step_size, dimensions)
        st.pyplot(fig_msd)

    with tab4:
        st.subheader("Sweep Analysis: Varying Decay Constant")
        st.markdown(r"Running a quick parameter sweep for $\lambda \in [0.05, 0.1, 0.2, 0.5, 0.8]$...")
        
        sweep_lambdas = [0.05, 0.1, 0.2, 0.5, 0.8]
        fig_sweep, ax_sweep = plt.subplots(figsize=(10, 6))
        
        # We reuse the same fixed parameters except for lambda
        for lam in sweep_lambdas:
            sweep_engine = MonteCarloEngine(
                num_particles=num_particles,
                dimensions=dimensions,
                step_size=step_size,
                decay_constant=lam,
                dt=dt,
                boundary_size=boundary_size,
                boundary_type=boundary_type,
                seed=42
            )
            sweep_res = sweep_engine.run(total_time)
            sweep_alive = sweep_res['alive_count'] / num_particles * 100
            ax_sweep.plot(sweep_res['time'], sweep_alive, label=f"$\\lambda$ = {lam}")
            
        ax_sweep.set_xlabel("Time")
        ax_sweep.set_ylabel("Survival Percentage (%)")
        ax_sweep.set_title("Response of Survival Curve to $\\lambda$")
        ax_sweep.legend()
        ax_sweep.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_sweep)

else:
    st.info("Configure parameters in the sidebar and click 'Run Simulation' to start.")
