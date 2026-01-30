# Usage Guide - Model Building Module

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Analysis

```bash
python main_model.py
```

This will execute all four modeling tasks and generate output CSV files.

## Expected Output

After running `main_model.py`, you should see:

```
======================================================================
         MCM 2026 PROBLEM C: DANCING WITH THE STARS ANALYSIS          
======================================================================

... detailed analysis for each task ...

Output Files Generated:
  ✓ task1_voting_estimates.csv
  ✓ task2_method_comparison.csv
  ✓ task3_factor_impacts.csv
  ✓ task3_industry_analysis.csv
  ✓ task3_age_analysis.csv
  ✓ task4_hybrid_system_results.csv
```

## Running Individual Tasks

You can also run individual tasks if needed:

### Task 1: Voting Estimation

```python
from data_preprocessing import DataPreprocessor
from task1_voting_estimation import run_task1_analysis

# Load data
preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
data = preprocessor.preprocess_all()

# Run Task 1
results = run_task1_analysis(data)

# Access results
print(f"Model R² Score: {results['metrics']['test_r2']:.4f}")
print(results['results'].head())
```

### Task 2: Ranking Comparison

```python
from data_preprocessing import DataPreprocessor
from task2_ranking_comparison import run_task2_analysis

preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
data = preprocessor.preprocess_all()

results = run_task2_analysis(data)
print(f"Fairer method: {results['fairness_metrics']['overall_fairness']['fairer_method']}")
```

### Task 3: Success Factors

```python
from data_preprocessing import DataPreprocessor
from task3_success_factors import run_task3_analysis

preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
data = preprocessor.preprocess_all()

results = run_task3_analysis(data)
print(results['factor_impacts'].head(10))
```

### Task 4: Improved Voting System

```python
from data_preprocessing import DataPreprocessor
from task4_improved_voting import run_task4_analysis

preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
data = preprocessor.preprocess_all()

results = run_task4_analysis(data)
print(f"System: {results['hybrid_system']['name']}")
```

## Output File Descriptions

### task1_voting_estimates.csv
Contains estimated audience votes for each contestant:
- `celebrity_name`: Name of contestant
- `season`: Season number
- `placement`: Actual placement
- `estimated_fan_votes_millions`: Predicted votes (millions)
- `lower_bound_95ci`: Lower 95% confidence interval
- `upper_bound_95ci`: Upper 95% confidence interval
- `confidence_interval_width`: Width of confidence interval

### task2_method_comparison.csv
Comparison of ranking vs percentage methods by season:
- `season`: Season number
- `num_contestants`: Number of contestants
- `correlation`: Correlation between methods
- `avg_rank_difference`: Average rank difference
- `major_disagreements`: Number of major disagreements
- `agreement_rate`: Rate of agreement

### task3_factor_impacts.csv
Quantified impact of each factor on success:
- `feature`: Feature name
- `importance`: Raw importance score
- `impact_percentage`: Impact as percentage
- `category`: Feature category

### task3_industry_analysis.csv
Success analysis by industry:
- `industry`: Industry/profession
- `count`: Number of contestants
- `success_count`: Number of successful contestants
- `success_rate`: Success rate
- `avg_age`: Average age
- `avg_placement`: Average placement

### task3_age_analysis.csv
Success analysis by age group:
- `age_group`: Age bracket
- `count`: Number of contestants
- `success_count`: Number of successful
- `success_rate`: Success rate
- `avg_placement`: Average placement

### task4_hybrid_system_results.csv
Results under proposed hybrid voting system:
- `celebrity_name`: Name
- `season`: Season number
- `placement`: Current placement
- `hybrid_score`: Score under hybrid system
- `hybrid_rank`: Rank under hybrid system
- `rank_change`: Change from current placement

## Interpreting Results

### Task 1: Vote Estimation
- Higher R² score (closer to 1.0) indicates better prediction accuracy
- Confidence intervals show uncertainty in estimates
- Feature importance reveals what drives votes

### Task 2: Fairness Analysis
- Higher correlation = more agreement between methods
- Lower rank differences = more consistent results
- Controversial contestants have large rank differences

### Task 3: Success Factors
- Higher impact percentage = stronger influence on success
- Performance metrics typically dominate
- Industry and age have secondary effects

### Task 4: Hybrid System
- Correlation with current system shows consistency
- Projected viewership increase from fairness improvements
- Rank changes show potential adjustments

## Troubleshooting

### Issue: Module not found
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: File not found
**Solution**: Ensure `2026_MCM_Problem_C_Data.csv` is in the same directory

### Issue: Memory error
**Solution**: The dataset is small (~400 rows), but if issues occur:
- Close other applications
- Use 64-bit Python
- Reduce cross-validation folds in code

### Issue: Warnings about convergence
**Solution**: These are normal and don't affect results. The models are optimized for accuracy over speed.

## Performance Notes

- Full analysis takes approximately 10-30 seconds
- Each task can be run independently if needed
- Output files are generated automatically
- All models use reproducible random seeds (42)

## Customization

### Adjust confidence level (Task 1)

In `task1_voting_estimation.py`, modify the `predict_with_confidence` call:

```python
predictions, lower, upper = estimator.predict_with_confidence(X, confidence_level=0.90)  # 90% instead of 95%
```

### Change success threshold (Task 3)

In `task3_success_factors.py`, modify:

```python
y = analyzer.create_success_labels(top_k=5)  # Top 5 instead of top 3
```

### Adjust hybrid system weights (Task 4)

In `task4_improved_voting.py`, modify the `design_hybrid_system` method weights:

```python
'judge_score': {'weight': 0.50, ...},  # Increase judge weight
'audience_vote': {'weight': 0.30, ...},  # Decrease audience weight
```

## Citation

If using this model for academic purposes:

```
MCM 2026 Problem C Model Building Module
Dancing with the Stars Analysis
Implementation: Python with scikit-learn, pandas, numpy, scipy
```

## Support

For questions about the implementation:
1. Check the README.md for overview
2. Review individual module docstrings
3. Examine the problem statement document
4. Review the generated output files

## Advanced Usage

### Access preprocessing components

```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
data = preprocessor.preprocess_all()

# Get specific feature types
features = preprocessor.get_feature_columns()
print(f"Industry features: {len(features['industry'])}")
print(f"Judge score features: {len(features['judge_scores'])}")
```

### Custom model training

```python
from task1_voting_estimation import VotingEstimator

estimator = VotingEstimator(data)
X, y = estimator.create_features()

# Train with custom parameters
from sklearn.ensemble import GradientBoostingRegressor
custom_model = GradientBoostingRegressor(n_estimators=200, max_depth=7)
custom_model.fit(X, y)
```

## License

This implementation is for academic purposes as part of MCM 2026.
