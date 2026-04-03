import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

class MLTrigger:
    """
    A Machine Learning-based trigger system that classifies events as Signal (1) or Background (0).
    """
    def __init__(self, model_type: str = 'logistic'):
        """
        Args:
            model_type: 'logistic', 'decision_tree', or 'random_forest'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.features = ['energy', 'momentum', 'noise_level']
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=5, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        self.fitted = False

    def fit(self, events: pd.DataFrame, labels: pd.Series):
        """
        Train the model on generated datasets.
        """
        X = events[self.features].values
        y = labels.values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def set_thresholds(self, probability_threshold: float):
        """
        Threshold on the probability output.
        """
        self.probability_threshold = probability_threshold

    def evaluate(self, events: pd.DataFrame) -> np.ndarray:
        """
        Evaluate events.
        """
        if not self.fitted:
            raise NotFittedError("The model must be trained before calling evaluate.")
            
        X = events[self.features].values
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self, 'probability_threshold'):
            probs = self.model.predict_proba(X_scaled)[:, 1] # Probability of being signal
            return (probs > self.probability_threshold).astype(int)
        else:
            return self.model.predict(X_scaled)
