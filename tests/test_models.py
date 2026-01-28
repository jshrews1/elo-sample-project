"""
Unit tests for ELO model and prediction functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.elo_model import ELOModel, AdaptiveELOModel
from src.data_prep import DataPreprocessor
from src.ensemble import PointDifferentialModel, WinLossModel, EnsemblePredictor


class TestELOModel:
    """Test cases for ELOModel class."""
    
    @pytest.fixture
    def elo(self):
        """Create a fresh ELO model for each test."""
        return ELOModel(initial_rating=1500, k_factor=32)
    
    def test_initialization(self, elo):
        """Test model initialization."""
        assert elo.initial_rating == 1500
        assert elo.k_factor == 32
        assert len(elo.ratings) == 0
    
    def test_initialize_team(self, elo):
        """Test initializing a single team."""
        elo.initialize_team('Duke')
        assert elo.get_rating('Duke') == 1500
    
    def test_initialize_teams(self, elo):
        """Test initializing multiple teams."""
        teams = ['Duke', 'Kansas', 'UCLA']
        elo.initialize_teams(teams)
        assert all(elo.get_rating(t) == 1500 for t in teams)
    
    def test_expected_score(self, elo):
        """Test expected win probability calculation."""
        # Equal ratings should give 50% probability
        prob = elo.expected_score(1500, 1500)
        assert abs(prob - 0.5) < 0.01
        
        # Higher rated team should have >50% probability
        prob_higher = elo.expected_score(1600, 1500)
        assert prob_higher > 0.5
    
    def test_update_rating(self, elo):
        """Test rating update after a game."""
        elo.initialize_teams(['Team A', 'Team B'])
        
        # Team A wins
        new_rating = elo.update_rating('Team A', 'Team B', 1.0)
        assert new_rating > 1500  # Rating should increase after win
    
    def test_process_game(self, elo):
        """Test processing a complete game."""
        elo.initialize_teams(['Duke', 'Kansas'])
        
        duke_rating, kansas_rating = elo.process_game('Duke', 'Kansas', 75, 72)
        
        assert duke_rating > 1500  # Duke won
        assert kansas_rating < 1500  # Kansas lost
    
    def test_predict_game(self, elo):
        """Test game prediction."""
        elo.initialize_teams(['Duke', 'Kansas'])
        
        prediction = elo.predict_game('Duke', 'Kansas')
        
        assert 'Duke' in prediction
        assert 'Kansas' in prediction
        assert abs(prediction['Duke'] + prediction['Kansas'] - 1.0) < 0.01


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_games(self):
        """Create sample game data."""
        return pd.DataFrame({
            'team_a': ['Duke', 'Kansas', 'UCLA', 'Duke'],
            'team_b': ['Kansas', 'UCLA', 'Duke', 'Kansas'],
            'score_a': [75, 80, 65, 72],
            'score_b': [72, 78, 68, 70]
        })
    
    def test_initialization(self, sample_games):
        """Test preprocessor initialization."""
        prep = DataPreprocessor(sample_games)
        assert len(prep.df) == 4
    
    def test_validate_data(self, sample_games):
        """Test data validation."""
        prep = DataPreprocessor(sample_games)
        assert prep.validate_data() is True
    
    def test_invalid_data(self):
        """Test validation with missing columns."""
        bad_df = pd.DataFrame({'team_a': ['Duke'], 'team_b': ['Kansas']})
        prep = DataPreprocessor(bad_df)
        
        with pytest.raises(ValueError):
            prep.validate_data()
    
    def test_feature_engineering(self, sample_games):
        """Test feature creation."""
        prep = DataPreprocessor(sample_games)
        prep.engineer_features()
        
        assert 'point_differential' in prep.df.columns
        assert 'win' in prep.df.columns
        assert 'margin' in prep.df.columns


class TestEnsemble:
    """Test cases for ensemble models."""
    
    @pytest.fixture
    def sample_games(self):
        """Create sample game data."""
        return pd.DataFrame({
            'team_a': ['Duke', 'Kansas', 'UCLA'] * 10,
            'team_b': ['Kansas', 'UCLA', 'Duke'] * 10,
            'score_a': np.random.randint(60, 90, 30),
            'score_b': np.random.randint(60, 90, 30)
        })
    
    def test_point_differential_model(self, sample_games):
        """Test point differential model."""
        model = PointDifferentialModel()
        model.fit(sample_games)
        
        prediction = model.predict('Duke', 'Kansas')
        assert 'Duke' in prediction
        assert 'Kansas' in prediction
        assert 0 <= prediction['Duke'] <= 1
    
    def test_win_loss_model(self, sample_games):
        """Test win-loss model."""
        model = WinLossModel()
        model.fit(sample_games)
        
        prediction = model.predict('Duke', 'Kansas')
        assert prediction['Duke'] + prediction['Kansas'] == pytest.approx(1.0)
    
    def test_ensemble_predictor(self, sample_games):
        """Test ensemble predictor."""
        ensemble = EnsemblePredictor()
        ensemble.add_model('test', PointDifferentialModel())
        ensemble.fit(sample_games)
        
        prediction = ensemble.predict('Duke', 'Kansas')
        assert 'ensemble' in prediction
        assert 'individual' in prediction


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
