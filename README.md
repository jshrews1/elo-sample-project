# Wharton High School Data Science Competition - ELO Rating Model Starter

A comprehensive starter project for implementing an ELO rating system to rank basketball teams and predict game outcomes. Built based on the [Wharton High School Data Science Competition Playbook](https://wsb.wharton.upenn.edu/wharton-data-competition/wharton-high-school-data-science-competition-playbook/).

## Project Overview

This project demonstrates how to build predictive models for basketball team ranking and game outcome prediction, following best practices from the Wharton competition. The competition challenges students to:

- **Analyze real-world basketball data** (5,000+ games)
- **Rank teams** using statistical and machine learning approaches
- **Predict game outcomes** with competing models
- **Create ensemble models** that combine multiple approaches for robustness

### Key Approaches from the Playbook

1. **ELO Rating System** - Dynamically rank teams based on game outcomes accounting for opponent strength
2. **Point Differential Model** - Use historical average margin of victory for predictions
3. **Win-Loss Analysis** - Track team win rates and historical performance
4. **Ensemble Methods** - Combine multiple models for more robust predictions
5. **Feature Engineering** - Create meaningful variables like efficiency metrics and strength of schedule

## Project Structure

```
elo-sample-project/
├── src/
│   ├── __init__.py
│   ├── elo_model.py           # ELO rating implementation
│   ├── data_prep.py           # Data cleaning and feature engineering
│   └── ensemble.py            # Multiple prediction models and ensemble
├── notebooks/
│   └── elo_basketball_analysis.ipynb  # Complete walkthrough example
├── data/
│   ├── sample_games.csv       # Example game data
│   ├── final_elo_rankings.csv # Output rankings
│   └── rating_history.csv     # Rating update history
├── tests/
│   └── test_models.py         # Unit tests (optional)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Getting Started

### Installation

1. **Clone or download this project**
   ```bash
   cd elo-sample-project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start (Command Line)

Run the complete example in seconds:

```bash
python quick_start.py
```

This script:
- Generates 50 sample basketball games with 6 teams
- Initializes an ELO model and processes all games
- Displays team statistics and current rankings
- Trains an ELO predictor model
- Fits an ensemble combining ELO, point differential, and win-loss models
- Shows example predictions and accuracy metrics

**Sample Output:**
```
Team Statistics:
              Team  Games  Wins  Win Rate  Avg Points  Point Diff
         Arizona      10    7     0.700       73.5        8.2
         Kansas      11    9     0.818       76.2       12.1
    ...

ELO Model Rankings (Top 5):
     Team  Rating
    Kansas    1621
  Arizona    1584
  ...

Ensemble Prediction Example:
Arizona vs Kansas
  Arizona: 24.1%
  Kansas: 75.9%
```

### Running the Example Notebook

1. **Launch Jupyter**
   ```bash
   jupyter notebook notebooks/elo_basketball_analysis.ipynb
   ```

   Or start the Jupyter server in the background:
   ```bash
   jupyter notebook
   ```
   Then navigate to `http://localhost:8888` and open the notebook

2. **Run the notebook cells sequentially**
   - The notebook walks through the complete workflow
   - Generates sample data and demonstrates all key techniques
   - Produces visualizations and exports results
   - **Important**: Run all cells from top to bottom in order; do not skip cells

3. **Expected Outputs**
   - Team statistics with win rates and efficiency metrics
   - ELO rankings visualization
   - Rating distribution analysis
   - CSV exports: `data/final_elo_rankings.csv` and `data/rating_history.csv`

## Core Components

### 1. ELO Model (`src/elo_model.py`)

The ELO rating system updates team ratings based on game outcomes using:

$$R_{new} = R_{old} + K \times (S - E)$$

Where:
- $R_{old}$ = Current ELO rating
- $K$ = K-factor (controls sensitivity, default 32)
- $S$ = Game result (1.0 = win, 0.5 = tie, 0.0 = loss)
- $E$ = Expected win probability: $E = \frac{1}{1 + 10^{(R_{opponent} - R_{team})/400}}$

**Key Classes:**
- `ELOModel` - Standard ELO implementation
- `AdaptiveELOModel` - ELO with adaptive K-factor for stabilizing top ratings

**Usage:**
```python
from src.elo_model import ELOModel

# Initialize model
elo = ELOModel(initial_rating=1500, k_factor=32)

# Initialize teams
elo.initialize_teams(['Duke', 'Kansas', 'UCLA', ...])

# Process a game
new_rating_a, new_rating_b = elo.process_game('Duke', 'Kansas', 75, 72)

# Get rankings
rankings = elo.get_rankings(top_n=10)

# Predict a game
prediction = elo.predict_game('Duke', 'Kansas')
```

### 2. Data Preparation (`src/data_prep.py`)

Handles data cleaning and feature engineering following Wharton playbook best practices:

- **Data Validation** - Check for required columns and data quality
- **Missing Value Handling** - Multiple strategies for incomplete data
- **Invalid Game Removal** - Filter out nonsensical games
- **Feature Engineering** - Create point differential, margins, win indicators
- **Team Statistics** - Calculate aggregate stats (win rate, avg points, etc.)
- **Possession Adjustment** - Normalize stats by pace of play
- **Division Handling** - Separate Division 1 and Division 2 teams

**Usage:**
```python
from src.data_prep import DataPreprocessor

# Load and prepare data
prep = DataPreprocessor(games_df)
prep.validate_data()
prep.handle_missing_values(strategy='drop')
prep.remove_invalid_games()
prep.engineer_features()

# Get team statistics
stats = prep.calculate_team_stats()
stats_df = prep.get_team_stats_dataframe()

# Split into train/test
train_df, test_df = prep.split_train_test(train_ratio=0.8)
```

### 3. Ensemble Prediction (`src/ensemble.py`)

Combines multiple modeling approaches for robust predictions:

- **Point Differential Model** - Predicts based on historical point margins
- **Win-Loss Model** - Predicts using historical win rates
- **ELO Model** - Incorporates opponent strength
- **Ensemble Predictor** - Aggregates predictions with custom weights

Following the Wharton approach, ensemble models often outperform individual models by capturing different patterns in the data.

**Usage:**
```python
from src.ensemble import create_default_ensemble

# Create ensemble with recommended models and weights
ensemble = create_default_ensemble(k_factor=32)

# Fit on training data
ensemble.fit(train_df)

# Make predictions
prediction = ensemble.predict('Duke', 'Kansas')

# Evaluate on test set
results = ensemble.evaluate(test_df)
print(f"Ensemble Accuracy: {results['ensemble_accuracy']:.2%}")
```

## Wharton Competition Insights

Based on the playbook analysis of winning submissions:

### Best Practices

1. **Start Simple** - Win-loss records and point differentials are surprisingly effective baselines
2. **Account for Context** - Strong models adjust for opponent strength (this is where ELO excels)
3. **Feature Engineering** - Create meaningful variables:
   - Points per possession (accounting for pace)
   - Offensive/Defensive efficiency
   - Home court advantage
   - Travel distance and rest days
4. **Use Multiple Models** - Ensemble methods reduce bias and variance
5. **Handle Division 2 Teams Carefully** - Either drop them or model separately
6. **Thoughtful Data Imputation** - Don't ignore missing data; address it case-by-case

### Model Comparison

The competition winning teams used:
- **First Place**: Multi-model ensemble combining consensus predictions
- **Second Place**: Machine learning approach for team ranking

All top teams incorporated:
- Strength of schedule adjustments
- Contextual factors (home/away, rest)
- Efficient use of feature engineering
- Ensemble or consensus methods

## Example Workflow

```python
import pandas as pd
from src.data_prep import DataPreprocessor
from src.ensemble import create_default_ensemble

# 1. Load your game data
games_df = pd.read_csv('data/basketball_games.csv')

# 2. Prepare data
prep = DataPreprocessor(games_df)
prep.validate_data()
prep.handle_missing_values('drop')
prep.remove_invalid_games()
prep.engineer_features()
games_df = prep.get_processed_data()

# 3. Split data
train_df, test_df = prep.split_train_test(0.8)

# 4. Build and train ensemble
ensemble = create_default_ensemble()
ensemble.fit(train_df)

# 5. Evaluate performance
results = ensemble.evaluate(test_df)
print(f"Accuracy: {results['ensemble_accuracy']:.2%}")

# 6. Make predictions
for team_a, team_b in [('Duke', 'Kansas'), ('UCLA', 'Gonzaga')]:
    pred = ensemble.predict(team_a, team_b)
    print(f"{team_a}: {pred['ensemble'][team_a]:.1%}")
```

## Customization Ideas

### Enhance the ELO Model

- **K-factor Adjustment**: Use higher K-factors during tournament play
- **Home Court Bonus**: Add bonus points for home teams
- **Rating Caps**: Limit extreme rating divergence
- **Recency Weighting**: Give more weight to recent games

### Add More Features

- Home/away splits
- Player injury status
- Travel distance and rest days
- Bench strength metrics
- Turnover rates and efficiency

### Advanced Models

- **Regression Models**: Linear/logistic regression with contextual features
- **Gradient Boosting**: XGBoost, LightGBM for non-linear patterns
- **Neural Networks**: Deep learning for complex pattern recognition
- **Bayesian Methods**: Account for uncertainty in ratings

### Prediction Improvements

- **Calibration**: Ensure probability predictions are well-calibrated
- **Cross-validation**: Use k-fold cross-validation for robust evaluation
- **Confidence Intervals**: Estimate prediction uncertainty
- **Win Probability Added (WPA)**: Measure incremental impact of predictions

## Key Metrics to Track

1. **Accuracy** - % of games predicted correctly
2. **Log Loss** - Measures probability calibration
3. **AUC-ROC** - Area under receiver operating characteristic curve
4. **Ranking Correlation** - Spearman correlation with true team strength
5. **Confidence Interval Calibration** - Do confidence intervals contain true value?

## Common Pitfalls to Avoid

1. **Data Leakage** - Don't use future information to predict past games
2. **Overfitting** - Validate on held-out test set; use cross-validation
3. **Class Imbalance** - Handle when one team consistently beats the other
4. **Missing Context** - Don't ignore important variables (home/away, rest, etc.)
5. **Stale Ratings** - Update ratings regularly; old data becomes irrelevant
6. **Ignoring Division Differences** - D2 teams don't predict D1 well; handle separately

## Resources

- **Wharton Competition**: https://wsb.wharton.upenn.edu/wharton-data-competition/
- **ELO System**: https://en.wikipedia.org/wiki/Elo_rating_system
- **Sports Analytics**: "The Sports Analytics Mindset" by Dean Oliver
- **Basketball Metrics**: Basketball-Reference advanced stats

## Files Description

### Main Source Files

| File | Purpose |
|------|---------|
| `src/elo_model.py` | ELO rating calculations and team ranking |
| `src/data_prep.py` | Data cleaning, validation, feature engineering |
| `src/ensemble.py` | Multiple prediction models and ensemble |
| `notebooks/elo_basketball_analysis.ipynb` | Complete working example |

### Output Files

| File | Description |
|------|-------------|
| `data/final_elo_rankings.csv` | Final team rankings with ratings |
| `data/rating_history.csv` | Complete history of rating updates |

## Next Steps

1. **Load Your Data** - Replace sample data with competition data
2. **Explore the Data** - Use `DataPreprocessor` to understand patterns
3. **Experiment with K-factors** - Test different sensitivity levels
4. **Add Features** - Implement possession adjustments, efficiency metrics
5. **Build Custom Models** - Create specialized models for your data
6. **Ensemble Creation** - Combine models with optimal weights
7. **Rigorous Testing** - Use cross-validation and held-out test sets
8. **Visualize Results** - Create compelling charts for presentation

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

See `requirements.txt` for exact versions.

## License

This starter project is provided as-is for educational purposes related to the Wharton High School Data Science Competition.

## Questions & Support

For questions about:
- **The Competition**: Visit https://wsb.wharton.upenn.edu/wharton-data-competition/
- **The Playbook**: Reference the official competition playbook
- **This Code**: Review docstrings and comments in source files

---

**Happy analyzing! Good luck with the Wharton High School Data Science Competition!** 🏀📊
