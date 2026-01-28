"""
Ensemble and Prediction Models

Combines multiple modeling approaches (ELO, point differential, etc.)
for robust game outcome predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Make this module work regardless of import style (relative or absolute)
__package__ = 'src'

# Import ELOModel classes (deferred to avoid circular imports)
# These will be imported when needed by create_default_ensemble
ELOModel = None
AdaptiveELOModel = None


class PointDifferentialModel:
    """
    Simple baseline model based on average point differential.
    
    Predicts outcomes based on historical average margin of victory.
    """
    
    def __init__(self):
        """Initialize the point differential model."""
        self.team_avg_differential = {}
    
    def fit(self, games_df: pd.DataFrame) -> None:
        """
        Fit the model on historical game data.
        
        Args:
            games_df: DataFrame with 'team_a', 'team_b', 'score_a', 'score_b'
        """
        self.team_avg_differential = {}
        
        # Collect all teams
        all_teams = set(games_df['team_a'].unique()) | set(games_df['team_b'].unique())
        
        for team in all_teams:
            home = games_df[games_df['team_a'] == team]
            away = games_df[games_df['team_b'] == team]
            
            home_diff = (home['score_a'] - home['score_b']).values
            away_diff = (away['score_b'] - away['score_a']).values
            
            all_diffs = np.concatenate([home_diff, away_diff])
            self.team_avg_differential[team] = np.mean(all_diffs) if len(all_diffs) > 0 else 0
    
    def predict(self, team_a: str, team_b: str) -> Dict[str, float]:
        """
        Predict win probability based on point differential.
        
        Args:
            team_a: First team
            team_b: Second team
            
        Returns:
            Dictionary with win probabilities
        """
        diff_a = self.team_avg_differential.get(team_a, 0)
        diff_b = self.team_avg_differential.get(team_b, 0)
        
        # Expected differential in the matchup
        expected_diff = diff_a - diff_b
        
        # Simple logistic conversion to probability
        # Higher differentials = higher win probability
        prob_a = 1 / (1 + np.exp(-expected_diff / 10))  # Scale factor of 10
        prob_b = 1 - prob_a
        
        return {team_a: prob_a, team_b: prob_b}


class WinLossModel:
    """
    Simple baseline model based on win-loss record.
    
    Predicts based on historical win rate.
    """
    
    def __init__(self):
        """Initialize the win-loss model."""
        self.win_rates = {}
    
    def fit(self, games_df: pd.DataFrame) -> None:
        """
        Fit the model on historical game data.
        
        Args:
            games_df: DataFrame with game data
        """
        self.win_rates = {}
        all_teams = set(games_df['team_a'].unique()) | set(games_df['team_b'].unique())
        
        for team in all_teams:
            home = games_df[games_df['team_a'] == team]
            away = games_df[games_df['team_b'] == team]
            
            home_wins = (home['score_a'] > home['score_b']).sum()
            away_wins = (away['score_b'] > away['score_a']).sum()
            
            total_games = len(home) + len(away)
            self.win_rates[team] = (home_wins + away_wins) / total_games if total_games > 0 else 0.5
    
    def predict(self, team_a: str, team_b: str) -> Dict[str, float]:
        """
        Predict win probability based on win rates.
        
        Args:
            team_a: First team
            team_b: Second team
            
        Returns:
            Dictionary with win probabilities
        """
        wr_a = self.win_rates.get(team_a, 0.5)
        wr_b = self.win_rates.get(team_b, 0.5)
        
        # Normalize win rates
        total = wr_a + wr_b
        if total == 0:
            return {team_a: 0.5, team_b: 0.5}
        
        prob_a = wr_a / total
        prob_b = wr_b / total
        
        return {team_a: prob_a, team_b: prob_b}


class EnsemblePredictor:
    """
    Ensemble model combining multiple prediction methods.
    
    Follows Wharton playbook approach of using multiple models for robustness.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            weights: Dictionary mapping model names to their weights.
                    If None, equal weights are used.
        """
        self.models = {}
        self.predictions = {}
        self.weights = weights
    
    def add_model(self, name: str, model) -> None:
        """
        Add a prediction model to the ensemble.
        
        Args:
            name: Name of the model (e.g., 'elo', 'point_diff')
            model: Model object with predict() method
        """
        self.models[name] = model
    
    def fit(self, games_df: pd.DataFrame) -> None:
        """
        Fit all models in the ensemble.
        
        Args:
            games_df: Training data
        """
        for name, model in self.models.items():
            print(f"Fitting {name} model...")
            
            # Special handling for ELOModel which uses process_game instead of fit
            model_class_name = model.__class__.__name__
            if model_class_name in ('ELOModel', 'AdaptiveELOModel'):
                # Initialize all teams
                all_teams = list(set(games_df['team_a'].unique()) | set(games_df['team_b'].unique()))
                model.initialize_teams(all_teams)
                
                # Process each game
                for _, row in games_df.iterrows():
                    model.process_game(row['team_a'], row['team_b'], row['score_a'], row['score_b'])
            else:
                # Standard fit method for other models
                model.fit(games_df)
    
    def predict(self, team_a: str, team_b: str) -> Dict:
        """
        Make ensemble prediction.
        
        Args:
            team_a: First team
            team_b: Second team
            
        Returns:
            Dictionary with ensemble probabilities and individual model predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models with add_model().")
        
        # Get predictions from all models
        all_predictions = {}
        for name, model in self.models.items():
            pred = model.predict(team_a, team_b)
            all_predictions[name] = pred
        
        # Calculate weights
        weights = self.weights
        if weights is None:
            # Equal weights
            weights = {name: 1 / len(self.models) for name in self.models.keys()}
        
        # Aggregate predictions
        ensemble_prob_a = sum(all_predictions[name][team_a] * weights.get(name, 1)
                             for name in self.models.keys())
        ensemble_prob_b = sum(all_predictions[name][team_b] * weights.get(name, 1)
                             for name in self.models.keys())
        
        # Normalize
        total = ensemble_prob_a + ensemble_prob_b
        ensemble_prob_a /= total
        ensemble_prob_b /= total
        
        return {
            'ensemble': {team_a: ensemble_prob_a, team_b: ensemble_prob_b},
            'individual': all_predictions,
            'weights': weights
        }
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate ensemble predictions on test data.
        
        Args:
            test_df: Test data with games to predict
            
        Returns:
            Dictionary with evaluation metrics
        """
        correct_ensemble = 0
        correct_by_model = {name: 0 for name in self.models.keys()}
        
        for idx, row in test_df.iterrows():
            team_a, team_b = row['team_a'], row['team_b']
            score_a, score_b = row['score_a'], row['score_b']
            actual_winner = team_a if score_a > score_b else team_b
            
            pred = self.predict(team_a, team_b)
            ensemble_winner = team_a if pred['ensemble'][team_a] > 0.5 else team_b
            
            if ensemble_winner == actual_winner:
                correct_ensemble += 1
            
            # Evaluate individual models
            for name, individual_pred in pred['individual'].items():
                model_winner = team_a if individual_pred[team_a] > 0.5 else team_b
                if model_winner == actual_winner:
                    correct_by_model[name] += 1
        
        total_games = len(test_df)
        
        results = {
            'ensemble_accuracy': correct_ensemble / total_games if total_games > 0 else 0,
            'individual_accuracies': {name: count / total_games for name, count in correct_by_model.items()},
            'total_test_games': total_games
        }
        
        return results


def create_default_ensemble(k_factor: float = 32) -> EnsemblePredictor:
    """
    Create a default ensemble with recommended models.
    
    Args:
        k_factor: K-factor for ELO model
        
    Returns:
        EnsemblePredictor with ELO, win-loss, and point differential models
    """
    # Import ELOModel here to avoid circular imports
    global ELOModel, AdaptiveELOModel
    if ELOModel is None:
        try:
            from elo_model import ELOModel as _ELOModel, AdaptiveELOModel as _AdaptiveELOModel
            ELOModel = _ELOModel
            AdaptiveELOModel = _AdaptiveELOModel
        except ImportError:
            from .elo_model import ELOModel as _ELOModel, AdaptiveELOModel as _AdaptiveELOModel
            ELOModel = _ELOModel
            AdaptiveELOModel = _AdaptiveELOModel
    
    ensemble = EnsemblePredictor(
        weights={
            'elo': 0.5,
            'point_differential': 0.3,
            'win_loss': 0.2
        }
    )
    
    # Add models
    elo = ELOModel(k_factor=k_factor)
    ensemble.add_model('elo', elo)
    
    ensemble.add_model('point_differential', PointDifferentialModel())
    ensemble.add_model('win_loss', WinLossModel())
    
    return ensemble
