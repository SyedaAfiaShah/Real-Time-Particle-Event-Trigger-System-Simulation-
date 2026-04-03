import numpy as np
import pandas as pd

class EventGenerator:
    """
    Generates synthetic particle physics events for testing trigger algorithms.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_events(
        self, 
        n_events: int = 10000, 
        signal_fraction: float = 0.05, 
        noise_scale: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate a set of mixed signal and background events.
        
        Args:
            n_events: Total number of events to generate.
            signal_fraction: Fraction of events that belong to true physics signals.
            noise_scale: Multiplier for background noise features.
            
        Returns:
            pd.DataFrame: Contains columns ['energy', 'momentum', 'noise_level', 'signal_label']
        """
        n_signal = int(n_events * signal_fraction)
        n_bkg = n_events - n_signal
        
        # --- Generate Background (Noise) Events ---
        # Background typically consists of low energy hits following an exponential decay,
        # with high variations in sensor noise.
        bkg_energy = self.rng.exponential(scale=15.0, size=n_bkg) 
        bkg_momentum = self.rng.exponential(scale=10.0, size=n_bkg)
        bkg_noise = self.rng.gamma(shape=5.0, scale=3.0 * noise_scale, size=n_bkg) # Long tail
        
        # --- Generate True Physics Signal Events ---
        # Signal typically comes from a physics process with a resonant peak (Gaussian),
        # with intrinsically lower random sensor noise.
        sig_energy = self.rng.normal(loc=60.0, scale=10.0, size=n_signal)
        # Apply lower bound to prevent unphysical negative energy
        sig_energy = np.clip(sig_energy, 0, None)
        
        sig_momentum = self.rng.normal(loc=45.0, scale=8.0, size=n_signal)
        sig_momentum = np.clip(sig_momentum, 0, None)
        
        # Signals might still have some detector noise, but usually cleaner
        sig_noise = self.rng.gamma(shape=2.0, scale=2.0 * noise_scale, size=n_signal)
        
        # Combine
        energy = np.concatenate([bkg_energy, sig_energy])
        momentum = np.concatenate([bkg_momentum, sig_momentum])
        noise_level = np.concatenate([bkg_noise, sig_noise])
        signal_label = np.concatenate([np.zeros(n_bkg, dtype=int), np.ones(n_signal, dtype=int)])
        
        df = pd.DataFrame({
            'energy': energy,
            'momentum': momentum,
            'noise_level': noise_level,
            'signal_label': signal_label
        })
        
        # Shuffle dataset
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        return df

if __name__ == "__main__":
    generator = EventGenerator()
    df = generator.generate_events(n_events=1000)
    print(df.head())
    print("\nValue counts:\n", df['signal_label'].value_counts())
