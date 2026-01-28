#!/usr/bin/env python3
"""
Simple test to verify ELO model imports and basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing ELO Model Installation...")
print("=" * 50)

try:
    print("\n1. Testing imports...")
    from src.elo_model import ELOModel, AdaptiveELOModel
    print("   ✓ ELOModel imported")
    
    from src.data_prep import DataPreprocessor
    print("   ✓ DataPreprocessor imported")
    
    from src.ensemble import create_default_ensemble, EnsemblePredictor
    print("   ✓ Ensemble models imported")
    
    print("\n2. Testing ELO Model...")
    elo = ELOModel(initial_rating=1500, k_factor=32)
    elo.initialize_teams(['Duke', 'Kansas', 'UCLA'])
    print("   ✓ ELO model created and teams initialized")
    
    print("\n3. Testing rating update...")
    new_rating = elo.update_rating('Duke', 'Kansas', 1.0)
    print(f"   ✓ Duke rating updated to {new_rating:.1f}")
    
    print("\n4. Testing prediction...")
    prediction = elo.predict_game('Duke', 'UCLA')
    print(f"   ✓ Prediction made: Duke {prediction['Duke']:.1%} vs UCLA {prediction['UCLA']:.1%}")
    
    print("\n5. Testing rankings...")
    rankings = elo.get_rankings()
    print("   ✓ Rankings retrieved:")
    print(f"\n{rankings.to_string(index=False)}")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! Project is ready to use.")
    print("\nNext steps:")
    print("  1. Run: jupyter notebook notebooks/elo_basketball_analysis.ipynb")
    print("  2. Or run: python quick_start.py")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
