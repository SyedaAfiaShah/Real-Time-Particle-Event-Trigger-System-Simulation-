import streamlit as st
import pandas as pd
import numpy as np

# Import our simulation modules
from src.event_generator import EventGenerator
from src.trigger_rules import RuleBasedTrigger
from src.statistical_trigger import StatisticalTrigger
from src.ml_trigger import MLTrigger
from src.metrics import TriggerMetrics
from src.visualization import TriggerVisualizer

st.set_page_config(page_title="Particle Trigger System", layout="wide")

# Inject Custom CSS for premium aesthetic
st.markdown("""
<style>
    /* Global Backgrounds and Fonts */
    .stApp {
        background-color: #050505;
        color: #e5e5e5;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 300;
        letter-spacing: -0.5px;
    }
    
    /* Elegant Transitions and Effects */
    .stButton > button {
        background: transparent !important;
        border: 1px solid #4ade80 !important;
        color: #4ade80 !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border-radius: 4px;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 13px !important;
    }
    .stButton > button:hover {
        background: rgba(74, 222, 128, 0.05) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(74, 222, 128, 0.15);
        border-color: #6ee7b7 !important;
        color: #6ee7b7 !important;
    }
    
    /* Metrics Styling - Minimalist */
    div[data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 200 !important;
        color: #ffffff !important;
        transition: color 0.4s ease, transform 0.4s ease;
    }
    div[data-testid="stMetricValue"]:hover {
        color: #4ade80 !important;
        transform: scale(1.02);
    }
    div[data-testid="stMetricLabel"] {
        color: #737373 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Sidebar Tweaks */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }

    /* Alert Boxes & Info Cards */
    .stAlert {
        border-radius: 2px;
        background: transparent !important;
        border: 1px solid #262626 !important;
        border-left: 3px solid #4ade80 !important;
        color: #a3a3a3 !important;
        transition: all 0.3s ease;
    }
    .stAlert:hover {
        border-color: #404040 !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Real-Time Particle Event Trigger System Simulation")
st.markdown("A research-grade simulation of data pipelines handling massive streams of particle events, comparing rule-based, statistical, and ML strategies.")

# --- Sidebar Controls ---
st.sidebar.header("Event Generation Parameters")
n_events = st.sidebar.number_input("Total Events", min_value=1000, max_value=50000, value=10000, step=1000)
signal_fraction = st.sidebar.slider("Signal Fraction", 0.01, 0.50, 0.05, 0.01, help="Percentage of true physics events.")
noise_scale = st.sidebar.slider("Noise Scale", 0.1, 5.0, 1.0, 0.1, help="Multiplier for background noise.")

st.sidebar.markdown("---")
st.sidebar.header("Trigger Configuration")
trigger_method = st.sidebar.radio(
    "Select Trigger Type",
    ["Rule-Based", "Statistical", "Machine Learning"]
)

# Render dynamic trigger controls depending on method
if trigger_method == "Rule-Based":
    st.sidebar.subheader("Threshold Parameters")
    energy_threshold = st.sidebar.slider("Energy Threshold", 10.0, 100.0, 40.0, 5.0)
    noise_threshold = st.sidebar.slider("Max Noise Threshold", 5.0, 50.0, 15.0, 1.0)
elif trigger_method == "Statistical":
    st.sidebar.subheader("Statistical Validation")
    p_value_threshold = st.sidebar.slider("P-Value Threshold", 0.001, 0.10, 0.05, 0.001, format="%.3f")
elif trigger_method == "Machine Learning":
    st.sidebar.subheader("ML Model Selection")
    ml_model_selected = st.sidebar.selectbox("Model Type", ["Logistic Regression", "Random Forest"])
    probability_threshold = st.sidebar.slider("Probability Threshold", 0.1, 0.9, 0.5, 0.05)


# Run Simulation Button
if st.sidebar.button("Run Simulation Pipeline", use_container_width=True):
    with st.spinner("Generating Events and Evaluating Trigger..."):
        # 1. Generate Data
        generator = EventGenerator(seed=42)
        df_events = generator.generate_events(n_events=n_events, signal_fraction=signal_fraction, noise_scale=noise_scale)
        
        # 2. Setup and Evaluate Trigger
        if trigger_method == "Rule-Based":
            trigger = RuleBasedTrigger(energy_threshold, noise_threshold)
            predictions = trigger.evaluate(df_events)
            
        elif trigger_method == "Statistical":
            trigger = StatisticalTrigger(p_value_threshold)
            # Need to fit on pure background for statistics
            df_bkg = df_events[df_events['signal_label'] == 0]
            trigger.fit(df_bkg)
            predictions = trigger.evaluate(df_events)
            
        elif trigger_method == "Machine Learning":
            model_map = {"Logistic Regression": "logistic", "Random Forest": "random_forest"}
            trigger = MLTrigger(model_type=model_map[ml_model_selected])
            # Train the ML model on 50% split (simplified here, train on the first half to eval the second)
            train_idx = int(n_events * 0.5)
            df_train = df_events.iloc[:train_idx]
            trigger.fit(df_train, df_train['signal_label'])
            trigger.set_thresholds(probability_threshold)
            
            # Predict the entire set for dashboard simplicity (or just val set)
            predictions = trigger.evaluate(df_events)

        # 3. Compute Metrics
        labels = df_events['signal_label'].values
        metrics = TriggerMetrics.evaluate(labels, predictions)
        
        # --- Visualization Generation ---
        viz = TriggerVisualizer()
        st.success(f"Simulation Complete! Filtered out {metrics['True Negatives']} background events.")
        
        # Metrics Top Row
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Signal Efficiency", f"{metrics['Efficiency']:.2%}")
        col2.metric("Background Rejection", f"{metrics['Background Rejection']:.2%}")
        col3.metric("Precision", f"{metrics['Precision']:.2%}")
        col4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
        
        st.markdown("---")
        
        # Visuals
        st.markdown("### Trigger Analysis Plots")
        
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            fig1 = viz.plot_feature_scatter(df_events)
            st.pyplot(fig1)
            
        with row1_col2:
            fig2 = viz.plot_energy_histogram(df_events, predictions)
            st.pyplot(fig2)
            
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            fig3 = viz.plot_trigger_decision(df_events, predictions)
            st.pyplot(fig3)
            
        with row2_col2:
            fig4 = viz.plot_confusion_matrix(labels, predictions)
            st.pyplot(fig4)
            
else:
    st.info("Adjust parameters and execute the pipeline to start.")
    
    st.markdown("""
    ### About the Trigger Data Pipeline
    In huge physics experiments like the LHC at CERN, detectors produce millions of collision events per second. We cannot store all of this data. A **Trigger System** rapidly analyzes features and rejects uninteresting background noise while retaining potential signal phenomena.
    
    You can explore three paradigms here:
    * **Rule-Based:** Hardcoded cuts (e.g., $E > 100$ AND noise $N < 2$). Fast, but often throws out edge-case real signals or accepts fake noise spikes.
    * **Statistical:** Uses outlier detection (z-scores/Likelihoods). Strong against known background shapes.
    * **Machine Learning:** Evaluates features holistically to build decision boundaries. High precision and efficiency, but requires compute resources for inference.
    """)
