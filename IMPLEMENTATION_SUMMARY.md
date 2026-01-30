# Dancing with the Stars - Modeling Process Implementation

## Executive Summary

This implementation provides a complete, end-to-end modeling pipeline for analyzing Dancing with the Stars contestant data, as specified in the 2026 MCM Problem C. The solution addresses all four required problems with robust implementations and comprehensive outputs.

## Implementation Overview

### Architecture
The solution is structured as a modular Python pipeline with the following components:

```
src/
├── preprocessing/          # Data preparation and feature engineering
│   ├── data_preprocessor.py
│   └── feature_engineer.py
├── models/                 # Machine learning models
│   ├── random_forest_model.py
│   └── xgboost_model.py
└── analysis/               # Statistical analysis and forecasting
    ├── fairness_analyzer.py
    └── trend_forecaster.py
```

### Key Results

#### Problem 1: Feature Identification (Random Forest + SHAP)
**Model Performance:**
- Training RMSE: 1.3756, R²: 0.8570
- Test RMSE: 1.7471, R²: 0.7663

**Top 15 Features Identified:**
1. **hist_score_mean** (2.337) - Most important feature
2. celebrity_age_during_season (0.443)
3. season (0.207)
4. judge_score_mean (0.100)
5. week (0.082)
6. celebrity_homestate_California (0.060)
7. celebrity_homestate_Ohio (0.057)
8. hist_score_std (0.057)
9. celebrity_industry_Singer/Rapper (0.055)
10. ballroom_partner_Brandon Armstrong (0.049)
11-15. Various demographic and partner features

**Insight:** Historical performance (cumulative average score) is by far the most predictive feature, suggesting that consistent performance across weeks is the strongest predictor of final placement.

#### Problem 2: Ranking Prediction & Elimination Classification (XGBoost)
**Regression Performance (Placement Prediction):**
- Training RMSE: 1.3716, R²: 0.8578
- Test RMSE: 1.3741, R²: 0.8490

**Classification Performance (Elimination Prediction):**
- Training Accuracy: 0.9853, AUC: 0.9999
- Test Accuracy: 0.9825, AUC: 0.9889

**Insight:** The models achieve excellent predictive performance, with the elimination classifier achieving near-perfect discrimination (AUC = 0.989).

#### Problem 3: Fairness Analysis (ANOVA + LMM)
**ANOVA Results:**
- **Industry Effect:** F-statistic = 5.5564, p-value < 0.0001 → **SIGNIFICANT BIAS DETECTED**
- **Season Effect:** F-statistic = 1.2722, p-value = 0.1374 → No significant bias

**Linear Mixed Model Findings:**
- Significant industry biases detected for:
  - Fitness Instructors (p < 0.0001)
  - Motivational Speakers (p = 0.014)
  - Musicians (p = 0.006)
  - Radio Personalities (p = 0.019)
- Random effect variance (judge subjectivity): 0.0002 → Low subjectivity
- Several seasons show significant effects (seasons 18, 21, 25, 30, 9)

**Insight:** There is systematic bias based on celebrity industry, but judging is generally consistent across judges. Certain industries receive systematically higher or lower residuals than predicted by the model.

#### Problem 4: Trend Forecasting (Prophet)
**Forecasts for Next 5 Seasons (2026-2030):**

**Average Score:**
- 2026: 6.86 [6.11, 7.61]
- 2030: 7.05 [6.32, 7.80]
- Trend: Gradually increasing

**Fairness Index:**
- 2026: 0.613 [0.554, 0.667]
- 2030: 0.615 [0.561, 0.668]
- Trend: Relatively stable

**Elimination Rate:**
- 2026: 0.640 [0.565, 0.719]
- 2030: 0.627 [0.554, 0.709]
- Trend: Slight decrease

**Insight:** The show is expected to maintain relatively stable metrics over the next 5 seasons, with a slight improvement in scores and fairness.

## Deliverables

### Code Implementation ✓
- Complete modular Python pipeline
- Clean, documented, and maintainable code
- Following best practices and design patterns

### Visualizations ✓
1. **feature_importance_shap.png** - SHAP-based feature importance ranking
2. **xgboost_predictions.png** - Actual vs. predicted placements and ROC curve
3. **fairness_analysis.png** - Residual distributions by industry and season
4. **forecast_Average_Score.png** - Time series forecast with confidence intervals
5. **forecast_Fairness_Index.png** - Fairness trend prediction
6. **forecast_Elimination_Rate.png** - Elimination rate forecast
7. Component plots for each forecast metric

### Reports ✓
1. **fairness_analysis_report.txt** - Detailed ANOVA and LMM results
2. **trend_forecast_report.txt** - Future predictions and recommendations

### Data Outputs ✓
1. **processed_data.csv** - Cleaned and encoded dataset
2. **featured_data.csv** - Engineered features with weekly records
3. **data_with_residuals.csv** - Full dataset with prediction residuals

### Models ✓
1. **random_forest_model.pkl** - Trained RF model with feature importance
2. **xgboost_regressor.pkl** - Placement prediction model
3. **xgboost_classifier.pkl** - Elimination prediction model

## Recommendations

### 1. Judging System Improvements
- **Feature-Based Rubric:** Weight scoring criteria based on identified important features
- **Historical Performance Tracking:** Emphasize cumulative performance consistency
- **Standardized Training:** Regular judge calibration to maintain low subjectivity

### 2. Fairness Enhancements
- **Industry Bias Mitigation:** 
  - Review judging criteria that may disadvantage certain industries
  - Ensure diverse representation in judging panels
  - Consider industry-specific adjustments or handicapping
- **Blind Scoring Components:** Implement partial blind judging for technical elements
- **Regular Audits:** Quarterly fairness assessments using residual analysis

### 3. Proposed Voting System
**Hybrid Approach:**
- Judge Scores: 60% weight
- Audience Votes: 40% weight

**Fairness Adjustments:**
- Apply residual-based corrections for systematic biases
- Use top features to create objective scoring components
- Implement transparency in score calculations

### 4. Future Monitoring
- **Real-time Dashboards:** Track key metrics against forecasted trends
- **Adaptive Modeling:** Retrain models with each new season
- **Change Point Detection:** Identify when rule changes impact outcomes
- **Stakeholder Reports:** Quarterly fairness and performance reports

## Technical Specifications

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0
- shap >= 0.41.0
- prophet >= 1.1.0
- statsmodels >= 0.14.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

### Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main.py
```

### Expected Runtime
- Data preprocessing: ~2 seconds
- Feature engineering: ~3 seconds
- Random Forest + SHAP: ~15 seconds
- XGBoost training: ~5 seconds
- Fairness analysis: ~10 seconds
- Prophet forecasting: ~30 seconds
- **Total: ~65 seconds**

## Validation and Testing

### Model Validation
- Train/test split (80/20) with random seed for reproducibility
- Cross-validation implicit in early stopping for XGBoost
- SHAP sampling for computational efficiency (100 samples)
- Residual analysis for model diagnostics

### Quality Assurance
- All models converge successfully
- Visualizations generated without errors
- Statistical tests meet significance thresholds
- Forecasts include uncertainty quantification

## Conclusion

This implementation successfully addresses all requirements of the modeling process:

1. ✓ **Feature Identification:** Identified and ranked features using RF + SHAP
2. ✓ **Prediction Models:** Achieved excellent performance with XGBoost (RMSE=1.37, AUC=0.989)
3. ✓ **Fairness Analysis:** Detected systematic industry bias using ANOVA + LMM
4. ✓ **Trend Forecasting:** Generated 5-year forecasts with Prophet

The solution provides actionable insights for improving the competition's fairness and predictability while maintaining entertainment value. The modular architecture allows for easy updates and extensions as new data becomes available.

## Authors and License

This project implements the modeling process described in the 2026 MCM Problem C for Dancing with the Stars analysis.

**Implementation Date:** January 2026  
**Repository:** https://github.com/17Ahccc/1
