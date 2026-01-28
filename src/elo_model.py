"""
ELO Rating System Implementation

This module implements the ELO rating system for ranking basketball teams.
Based on Wharton High School Data Science Competition playbook approach.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


class ELOModel:
    """
    ELO rating system for ranking teams based on game outcomes.
    
    The ELO system dynamically updates team ratings based on wins/losses
    and accounts for the strength of the opponent.
    """
    
    def __init__(self, initial_rating: float = 1500, k_factor: float = 32):
        """
        Initialize ELO model parameters.
        
        Args:
            initial_rating: Starting ELO rating for all teams (default 1500)
            k_factor: Controls how much ratings change per game (default 32)
                     Higher K = faster rating changes
                     Lower K = more stable ratings
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.history: List[Dict] = []
    
    def initialize_team(self, team: str, rating: float = None) -> None:
        """Initialize a team with an ELO rating."""
        if rating is None:
            rating = self.initial_rating
        self.ratings[team] = rating
    
    def initialize_teams(self, teams: List[str]) -> None:
        """Initialize multiple teams with default ELO rating."""
        for team in teams:
            self.initialize_team(team)
    
    def get_rating(self, team: str) -> float:
        """Get current ELO rating for a team."""
        if team not in self.ratings:
            self.initialize_team(team)
        return self.ratings[team]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected win probability for team A against team B.
        
        Uses the standard ELO formula: P_A = 1 / (1 + 10^((R_B - R_A)/400))
        
        Args:
            rating_a: ELO rating of team A
            rating_b: ELO rating of team B
            
        Returns:
            Probability that team A wins (0 to 1)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_rating(self, team: str, opponent: str, result: float, 
                     k_factor: float = None) -> float:
        """
        Update a team's ELO rating after a game.
        
        Args:
            team: Team whose rating is being updated
            opponent: Opposing team
            result: Game result for the team (1.0 = win, 0.5 = tie, 0.0 = loss)
            k_factor: Optional custom K-factor for this update
            
        Returns:
            New ELO rating for the team
        """
        if k_factor is None:
            k_factor = self.k_factor
        
        rating_team = self.get_rating(team)
        rating_opponent = self.get_rating(opponent)
        
        expected = self.expected_score(rating_team, rating_opponent)
        new_rating = rating_team + k_factor * (result - expected)
        
        self.ratings[team] = new_rating
        
        # Record the update
        self.history.append({
            'team': team,
            'opponent': opponent,
            'result': result,
            'old_rating': rating_team,
            'new_rating': new_rating,
            'expected_score': expected,
            'rating_change': new_rating - rating_team
        })
        
        return new_rating
    
    def process_game(self, team_a: str, team_b: str, 
                    score_a: float, score_b: float) -> Tuple[float, float]:
        """
        Process a game between two teams and update both ratings.
        
        Args:
            team_a: First team
            team_b: Second team
            score_a: Points scored by team A
            score_b: Points scored by team B
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        # Determine result (1 = win, 0 = loss)
        result_a = 1.0 if score_a > score_b else (0.5 if score_a == score_b else 0.0)
        result_b = 1.0 - result_a if result_a != 0.5 else 0.5
        
        # Update both teams
        new_rating_a = self.update_rating(team_a, team_b, result_a)
        new_rating_b = self.update_rating(team_b, team_a, result_b)
        
        return new_rating_a, new_rating_b
    
    def predict_game(self, team_a: str, team_b: str) -> Dict[str, float]:
        """
        Predict the outcome of a game between two teams.
        
        Args:
            team_a: First team
            team_b: Second team
            
        Returns:
            Dictionary with win probabilities for both teams
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        prob_a = self.expected_score(rating_a, rating_b)
        prob_b = 1 - prob_a
        
        return {
            team_a: prob_a,
            team_b: prob_b,
            'rating_diff': rating_a - rating_b
        }
    
    def predict(self, team_a: str, team_b: str) -> Dict[str, float]:
        """
        Alias for predict_game() to maintain API consistency with other models.
        
        Args:
            team_a: First team
            team_b: Second team
            
        Returns:
            Dictionary with win probabilities for both teams
        """
        return self.predict_game(team_a, team_b)
    
    def get_rankings(self, top_n: int = None) -> pd.DataFrame:
        """
        Get team rankings sorted by ELO rating.
        
        Args:
            top_n: Return only top N teams (None = all teams)
            
        Returns:
            DataFrame with team name, rating, and rank
        """
        rankings = pd.DataFrame([
            {'team': team, 'rating': rating}
            for team, rating in self.ratings.items()
        ]).sort_values('rating', ascending=False).reset_index(drop=True)
        
        rankings['rank'] = range(1, len(rankings) + 1)
        
        if top_n:
            rankings = rankings.head(top_n)
        
        return rankings[['rank', 'team', 'rating']]
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Get full history of rating updates as a DataFrame."""
        return pd.DataFrame(self.history)


class AdaptiveELOModel(ELOModel):
    """
    Extended ELO model with adaptive K-factor based on rating difference.
    
    Teams with higher ratings use lower K-factors to stabilize top rankings.
    This is commonly used in professional rating systems.
    """
    
    def __init__(self, initial_rating: float = 1500, base_k_factor: float = 32,
                 high_rating_threshold: float = 2000, high_rating_k: float = 16):
        """
        Initialize Adaptive ELO model.
        
        Args:
            initial_rating: Starting rating for all teams
            base_k_factor: K-factor for lower-rated teams
            high_rating_threshold: Rating above which reduced K-factor applies
            high_rating_k: K-factor for high-rated teams
        """
        super().__init__(initial_rating, base_k_factor)
        self.high_rating_threshold = high_rating_threshold
        self.high_rating_k = high_rating_k
    
    def get_k_factor(self, rating: float) -> float:
        """
        Get adaptive K-factor based on current rating.
        
        Higher-rated teams have lower K-factors for stability.
        """
        if rating >= self.high_rating_threshold:
            return self.high_rating_k
        return self.k_factor
    
    def update_rating(self, team: str, opponent: str, result: float, 
                     k_factor: float = None) -> float:
        """Update rating using adaptive K-factor."""
        if k_factor is None:
            rating = self.get_rating(team)
            k_factor = self.get_k_factor(rating)
        
        return super().update_rating(team, opponent, result, k_factor)
