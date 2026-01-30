# Dancing with the Stars - Modeling Process

This project implements a comprehensive modeling pipeline to analyze contestant performance, predict rankings and eliminations, analyze fairness, and forecast future trends for the Dancing with the Stars competition.

## Project Structure

```
.
├── data/                           # Data directory
├── src/                           # Source code
│   ├── preprocessing/             # Data preprocessing modules
│   ├── models/                    # Model training modules
│   ├── analysis/                  # Analysis modules (ANOVA, LMM)
│   └── visualization/             # Visualization utilities
├── notebooks/                     # Jupyter notebooks for exploration
├── visualizations/                # Generated plots and charts
├── reports/                       # Generated reports
├── requirements.txt               # Python dependencies
└── main.py                        # Main execution script

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete modeling pipeline:

```bash
python main.py
```

## Modeling Process

### Problem 1: Feature Identification (Random Forest + SHAP)
- Data preprocessing with missing value imputation
- Feature engineering (mean/std scores, historical features)
- Random Forest model training
- SHAP value calculation and feature importance ranking

### Problem 2: Ranking Prediction & Elimination Classification (XGBoost)
- XGBoost regression for placement prediction
- XGBoost classification for elimination prediction
- Residual calculation for fairness analysis

### Problem 3: Fairness Analysis (ANOVA + LMM)
- ANOVA testing for structural biases
- Linear Mixed Models for random effects analysis
- Identification of systematic unfairness

### Problem 4: Trend Forecasting (Prophet)
- Time series aggregation by season
- Prophet model with change points
- Future trend forecasting (3-5 seasons)

## Output

- Feature importance plots (SHAP)
- Model performance metrics (RMSE, AUC)
- Fairness analysis results
- Trend forecast visualizations
- Comprehensive PDF report

## Authors

This project implements the modeling process described in the 2026 MCM Problem C.
