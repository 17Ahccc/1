# Quick Start Guide

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/17Ahccc/1.git
cd 1
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Complete Pipeline (Recommended)
Run all four problems in sequence:
```bash
python main.py
```

This will:
- Preprocess data and engineer features
- Train Random Forest model and calculate SHAP values
- Train XGBoost models for prediction
- Perform fairness analysis (ANOVA + LMM)
- Generate trend forecasts with Prophet
- Create all visualizations and reports

**Expected runtime:** ~65 seconds

### Individual Modules

You can also run individual modules for specific analyses:

#### Problem 1: Feature Identification
```python
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.random_forest_model import RandomForestFeatureSelector

# Preprocess data
preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
processed_df = preprocessor.preprocess()

# Engineer features
engineer = FeatureEngineer()
featured_df = engineer.create_features(processed_df)

# Train Random Forest and get feature importance
rf_model = RandomForestFeatureSelector(task='regression')
rf_model.prepare_data(featured_df, target_col='placement')
rf_model.train()
rf_model.calculate_shap_values(sample_size=100)
feature_importance = rf_model.rank_features(top_k=15)
```

#### Problem 2: XGBoost Prediction
```python
from src.models.xgboost_model import XGBoostMultiTask

# Using top features from Problem 1
xgb_model = XGBoostMultiTask()
xgb_model.prepare_data(featured_df, feature_cols=top_features)
xgb_model.train_regression()
xgb_model.train_classification()
df_with_residuals = xgb_model.calculate_residuals(featured_df, top_features)
```

#### Problem 3: Fairness Analysis
```python
from src.analysis.fairness_analyzer import FairnessAnalyzer

analyzer = FairnessAnalyzer()
fairness_df = analyzer.prepare_fairness_data(df_with_residuals)
analyzer.perform_anova_industry(fairness_df)
analyzer.perform_anova_season(fairness_df)
analyzer.fit_linear_mixed_model(fairness_df)
```

#### Problem 4: Trend Forecasting
```python
from src.analysis.trend_forecaster import TrendForecaster

forecaster = TrendForecaster()
season_metrics = forecaster.aggregate_by_season(df_with_residuals)

# Forecast a specific metric
ts_data = forecaster.prepare_time_series(season_metrics, 'avg_score')
model = forecaster.train_prophet_model(ts_data, 'Average Score')
forecast = forecaster.forecast_future('Average Score', periods=5)
```

## Output Files

After running the pipeline, you'll find:

### Visualizations (`visualizations/`)
- `feature_importance_shap.png` - Top 15 features by SHAP importance
- `xgboost_predictions.png` - Model prediction quality
- `fairness_analysis.png` - Residual distributions
- `forecast_*.png` - Trend forecasts for each metric

### Reports (`reports/`)
- `fairness_analysis_report.txt` - ANOVA and LMM results
- `trend_forecast_report.txt` - Future predictions

### Data (`data/`)
- `processed_data.csv` - Cleaned data
- `featured_data.csv` - Engineered features
- `data_with_residuals.csv` - Data with predictions and residuals

### Models (`models/`)
- `random_forest_model.pkl` - RF model with feature importance
- `xgboost_regressor.pkl` - Placement prediction model
- `xgboost_classifier.pkl` - Elimination prediction model

## Viewing Results

### Visualizations
Open any PNG file in `visualizations/` directory:
```bash
# On Linux/Mac
open visualizations/feature_importance_shap.png

# On Windows
start visualizations\feature_importance_shap.png
```

### Reports
View text reports:
```bash
cat reports/fairness_analysis_report.txt
cat reports/trend_forecast_report.txt
```

## Troubleshooting

### ImportError: No module named 'prophet'
```bash
pip install prophet
```

### SHAP computation is slow
Reduce sample size in `main.py`:
```python
rf_model.calculate_shap_values(sample_size=50)  # Default is 100
```

### Prophet warnings about convergence
These are normal and don't affect the quality of forecasts. You can suppress them:
```python
import warnings
warnings.filterwarnings('ignore')
```

### Memory issues
If you run into memory problems, reduce the dataset or use fewer features:
```python
# In main.py, reduce top_k
feature_importance = rf_model.rank_features(top_k=10)  # Default is 15
```

## Next Steps

1. **Explore the data:** Open `data/featured_data.csv` in your favorite tool
2. **Analyze features:** Review `visualizations/feature_importance_shap.png`
3. **Check fairness:** Read `reports/fairness_analysis_report.txt`
4. **View forecasts:** Open forecast visualizations
5. **Customize:** Modify parameters in `main.py` for your specific needs

## Key Findings

- **Most Important Feature:** Historical cumulative score mean
- **Model Performance:** RMSE=1.37 (placement), AUC=0.989 (elimination)
- **Fairness Issue:** Industry bias detected (p<0.0001)
- **Forecast:** Stable trends expected over next 5 seasons

## Support

For issues or questions:
1. Check `IMPLEMENTATION_SUMMARY.md` for detailed documentation
2. Review the code comments in `src/` modules
3. Ensure all dependencies are correctly installed

## Citation

If you use this implementation, please reference:
```
Dancing with the Stars Modeling Process Implementation
Problem: 2026 MCM Problem C
Repository: https://github.com/17Ahccc/1
```
