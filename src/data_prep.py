"""
Data Preparation Module

Utilities for cleaning, preparing, and exploring basketball game data.
Follows best practices from Wharton competition playbook.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict


class DataPreprocessor:
    """Handle data cleaning and feature engineering for basketball games."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize preprocessor with game data.
        
        Args:
            df: DataFrame with columns: 'team_a', 'team_b', 'score_a', 'score_b', 
                and optionally 'date', 'location', 'division'
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.stats = {}
    
    def validate_data(self) -> bool:
        """
        Validate that required columns exist.
        
        Returns:
            True if data is valid, raises error otherwise
        """
        required_cols = ['team_a', 'team_b', 'score_a', 'score_b']
        missing = [col for col in required_cols if col not in self.df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return True
    
    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: 'drop' to remove rows with NaN, 'mean' to fill numeric cols
            
        Returns:
            Cleaned DataFrame
        """
        if strategy == 'drop':
            initial_len = len(self.df)
            self.df = self.df.dropna()
            dropped = initial_len - len(self.df)
            print(f"Dropped {dropped} rows with missing values")
        
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(
                self.df[numeric_cols].mean()
            )
        
        return self.df
    
    def remove_invalid_games(self) -> pd.DataFrame:
        """
        Remove games where scores don't make sense (e.g., both teams 0).
        
        Returns:
            Cleaned DataFrame
        """
        initial_len = len(self.df)
        
        # Remove games where both teams scored 0
        self.df = self.df[~((self.df['score_a'] == 0) & (self.df['score_b'] == 0))]
        
        # Remove games where team played itself
        self.df = self.df[self.df['team_a'] != self.df['team_b']]
        
        removed = initial_len - len(self.df)
        print(f"Removed {removed} invalid games")
        
        return self.df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create new features for modeling.
        
        Features created:
        - point_differential: Score difference (positive for team_a win)
        - game_total_score: Sum of both teams' scores
        - win: Binary indicator of win for team_a
        - margin: Absolute score difference
        
        Returns:
            DataFrame with new features
        """
        self.df['point_differential'] = self.df['score_a'] - self.df['score_b']
        self.df['game_total_score'] = self.df['score_a'] + self.df['score_b']
        self.df['win'] = (self.df['score_a'] > self.df['score_b']).astype(int)
        self.df['margin'] = np.abs(self.df['point_differential'])
        
        return self.df
    
    def calculate_team_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate aggregate statistics for each team.
        
        Returns:
            Dictionary of team stats including win rate, point differential, etc.
        """
        stats = {}
        
        # Collect all games for each team
        all_teams = set(self.df['team_a'].unique()) | set(self.df['team_b'].unique())
        
        for team in all_teams:
            home_games = self.df[self.df['team_a'] == team]
            away_games = self.df[self.df['team_b'] == team]
            
            all_games = len(home_games) + len(away_games)
            wins = len(home_games[home_games['score_a'] > home_games['score_b']]) + \
                   len(away_games[away_games['score_b'] > away_games['score_a']])
            
            avg_points_for = (home_games['score_a'].sum() + away_games['score_b'].sum()) / all_games if all_games > 0 else 0
            avg_points_against = (home_games['score_b'].sum() + away_games['score_a'].sum()) / all_games if all_games > 0 else 0
            
            stats[team] = {
                'games': all_games,
                'wins': wins,
                'losses': all_games - wins,
                'win_rate': wins / all_games if all_games > 0 else 0,
                'avg_points_for': avg_points_for,
                'avg_points_against': avg_points_against,
                'avg_point_differential': avg_points_for - avg_points_against,
                'strength_of_schedule': 0  # Placeholder, calculated separately
            }
        
        self.stats = stats
        return stats
    
    def filter_division(self, division: str = 'D1') -> pd.DataFrame:
        """
        Filter games to a specific division.
        
        Args:
            division: Division to filter ('D1', 'D2', etc.)
            
        Returns:
            Filtered DataFrame
        """
        if 'division' not in self.df.columns:
            print("Warning: 'division' column not found")
            return self.df
        
        initial_len = len(self.df)
        self.df = self.df[self.df['division'] == division]
        print(f"Filtered to {len(self.df)} games in division {division}")
        
        return self.df
    
    def get_team_stats_dataframe(self) -> pd.DataFrame:
        """
        Convert team stats to a DataFrame sorted by wins.
        
        Returns:
            DataFrame with team statistics
        """
        if not self.stats:
            self.calculate_team_stats()
        
        stats_df = pd.DataFrame.from_dict(self.stats, orient='index')
        stats_df['team'] = stats_df.index
        stats_df = stats_df.sort_values('wins', ascending=False)
        
        return stats_df[['team', 'games', 'wins', 'losses', 'win_rate', 
                        'avg_points_for', 'avg_points_against', 'avg_point_differential']]
    
    def split_train_test(self, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Args:
            train_ratio: Proportion of data for training (default 0.8)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(self.df) * train_ratio)
        train_df = self.df.iloc[:split_idx].copy()
        test_df = self.df.iloc[split_idx:].copy()
        
        print(f"Train: {len(train_df)} games | Test: {len(test_df)} games")
        
        return train_df, test_df
    
    def get_processed_data(self) -> pd.DataFrame:
        """Get the current processed DataFrame."""
        return self.df.copy()
    
    def summary_report(self) -> Dict:
        """
        Generate a summary report of the data.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'total_games': len(self.df),
            'unique_teams': len(set(self.df['team_a'].unique()) | set(self.df['team_b'].unique())),
            'avg_points_per_game': (self.df['score_a'] + self.df['score_b']).mean(),
            'max_score': max(self.df['score_a'].max(), self.df['score_b'].max()),
            'min_score': min(self.df['score_a'].min(), self.df['score_b'].min()),
            'avg_margin': self.df['margin'].mean() if 'margin' in self.df.columns else None,
        }


class PossessionAdjustmentCalculator:
    """
    Calculate possession-adjusted efficiency metrics.
    
    Following Wharton playbook approach: normalize stats by pace of play.
    """
    
    @staticmethod
    def calculate_points_per_possession(games_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate points per possession for each team.
        
        Args:
            games_df: DataFrame with game data
            
        Returns:
            Dictionary mapping team names to PPP
        """
        ppp = {}
        all_teams = set(games_df['team_a'].unique()) | set(games_df['team_b'].unique())
        
        for team in all_teams:
            # Simplified: use total points / total games as proxy
            # In real analysis, would need FGA/FTA data to calculate possessions
            home = games_df[games_df['team_a'] == team]
            away = games_df[games_df['team_b'] == team]
            
            total_points = home['score_a'].sum() + away['score_b'].sum()
            total_games = len(home) + len(away)
            
            ppp[team] = total_points / total_games if total_games > 0 else 0
        
        return ppp
    
    @staticmethod
    def calculate_defensive_efficiency(games_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate points allowed per game (defensive efficiency proxy).
        
        Args:
            games_df: DataFrame with game data
            
        Returns:
            Dictionary mapping team names to defensive efficiency
        """
        def_eff = {}
        all_teams = set(games_df['team_a'].unique()) | set(games_df['team_b'].unique())
        
        for team in all_teams:
            home = games_df[games_df['team_a'] == team]
            away = games_df[games_df['team_b'] == team]
            
            total_allowed = home['score_b'].sum() + away['score_a'].sum()
            total_games = len(home) + len(away)
            
            def_eff[team] = total_allowed / total_games if total_games > 0 else 0
        
        return def_eff
