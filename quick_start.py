"""
Example: Quick Start with Sample Data

This script demonstrates basic usage of the ELO model with sample data.
Run this to verify everything is set up correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.elo_model import ELOModel
from src.data_prep import DataPreprocessor
from src.ensemble import create_default_ensemble


def generate_sample_data(n_games=50):
    """Generate sample basketball game data."""
    np.random.seed(42)
    
    teams = ['Duke', 'North Carolina', 'Kansas', 'UCLA', 'Indiana', 'Arizona']
    games = []
    start_date = datetime(2024, 11, 1)
    
    for i in range(n_games):
        team_a = np.random.choice(teams)
        team_b = np.random.choice([t for t in teams if t != team_a])
        
        score_a = int(np.random.normal(75, 8))
        score_b = int(np.random.normal(72, 8))
        
        games.append({
            'date': start_date + timedelta(days=i),
            'team_a': team_a,
            'team_b': team_b,
            'score_a': max(50, score_a),
            'score_b': max(50, score_b)
        })
    
    return pd.DataFrame(games).sort_values('date').reset_index(drop=True)


def main():
    """Main example workflow."""
    print("=" * 60)
    print("Wharton ELO Model - Quick Start Example")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n1. Generating sample basketball game data...")
    games_df = generate_sample_data(n_games=50)
    print(f"   ✓ Generated {len(games_df)} games")
    print(f"   Date range: {games_df['date'].min().date()} to {games_df['date'].max().date()}")
    
    # 2. Prepare data
    print("\n2. Preparing and exploring data...")
    prep = DataPreprocessor(games_df)
    prep.validate_data()
    prep.engineer_features()
    team_stats = prep.calculate_team_stats()
    print(f"   ✓ Found {len(team_stats)} unique teams")
    
    print("\n   Team Statistics:")
    stats_df = prep.get_team_stats_dataframe()
    print(stats_df.to_string(index=False))
    
    # 3. Split data
    print("\n3. Splitting into train/test sets...")
    train_df, test_df = prep.split_train_test(train_ratio=0.7)
    
    # 4. Simple ELO example
    print("\n4. Running ELO Model...")
    elo = ELOModel(initial_rating=1500, k_factor=32)
    elo.initialize_teams(list(team_stats.keys()))
    
    # Process training data
    for _, row in train_df.iterrows():
        elo.process_game(row['team_a'], row['team_b'], row['score_a'], row['score_b'])
    
    print("   ✓ Processed all training games")
    
    # Show rankings
    rankings = elo.get_rankings()
    print("\n   Final ELO Rankings:")
    print(rankings.to_string(index=False))
    
    # 5. Make predictions on test set
    print("\n5. Making predictions on test set...")
    correct = 0
    for _, row in test_df.iterrows():
        team_a, team_b = row['team_a'], row['team_b']
        actual_winner = team_a if row['score_a'] > row['score_b'] else team_b
        
        pred = elo.predict_game(team_a, team_b)
        pred_winner = team_a if pred[team_a] > 0.5 else team_b
        
        if pred_winner == actual_winner:
            correct += 1
    
    accuracy = correct / len(test_df) if len(test_df) > 0 else 0
    print(f"   ✓ Test set accuracy: {accuracy:.1%} ({correct}/{len(test_df)})")
    
    # 6. Ensemble predictions
    print("\n6. Running Ensemble Model...")
    ensemble = create_default_ensemble(k_factor=32)
    ensemble.fit(train_df)
    
    results = ensemble.evaluate(test_df)
    print(f"   ✓ Ensemble accuracy: {results['ensemble_accuracy']:.1%}")
    
    print("\n   Individual model accuracies:")
    for model_name, acc in results['individual_accuracies'].items():
        print(f"      {model_name}: {acc:.1%}")
    
    # 7. Example prediction
    print("\n7. Example Match Prediction:")
    if len(team_stats) >= 2:
        team_a = list(team_stats.keys())[0]
        team_b = list(team_stats.keys())[1]
        
        pred = ensemble.predict(team_a, team_b)
        print(f"\n   {team_a} vs {team_b}:")
        print(f"      {team_a}: {pred['ensemble'][team_a]:.1%}")
        print(f"      {team_b}: {pred['ensemble'][team_b]:.1%}")
    
    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Load your own basketball game data")
    print("2. Run the Jupyter notebook: jupyter notebook notebooks/")
    print("3. Experiment with different K-factors and model weights")
    print("4. Add more features (possession-adjusted stats, etc.)")
    print("\nFor more information, see README.md")


if __name__ == '__main__':
    main()
