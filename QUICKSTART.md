# MCM 2026 Problem C - Quick Start Guide

## 快速开始指南

本指南帮助您快速开始使用MCM数据分析框架。

## 步骤 1: 环境配置

### 安装依赖包

```bash
pip install -r requirements.txt
```

或者手动安装主要依赖：

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn xgboost lightgbm statsmodels plotly jupyter
```

## 步骤 2: 运行分析

### 方法 A: 运行完整分析流程（推荐）

一次性运行所有分析模块：

```bash
python main.py --all
```

这将执行：
1. 数据预处理
2. 探索性数据分析
3. 统计建模
4. 预测建模
5. 模型评估
6. 数据可视化

预计耗时：5-10分钟

### 方法 B: 分步运行

如果你想单独运行某个模块：

```bash
# 仅数据预处理
python main.py --preprocess

# 仅探索性数据分析
python main.py --eda

# 仅统计建模
python main.py --statistical

# 仅预测建模
python main.py --predict

# 仅模型评估
python main.py --evaluate

# 仅可视化
python main.py --visualize
```

### 方法 C: 在Python中逐步运行

```python
# 1. 数据预处理
from src.data_preprocessing import DWTSDataPreprocessor

preprocessor = DWTSDataPreprocessor()
processed_data = preprocessor.process()
preprocessor.save_processed_data()

# 2. 探索性数据分析
from src.exploratory_analysis import EDAnalyzer

analyzer = EDAnalyzer()
analyzer.run_full_analysis()

# 3. 统计建模
from src.statistical_models import StatisticalModeler

modeler = StatisticalModeler()
modeler.run_all_models()

# 4. 预测建模
from src.prediction_models import PredictionModeler

predictor = PredictionModeler()
predictor.run_all_models()

# 5. 模型评估
from src.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.run_full_evaluation()

# 6. 可视化
from src.visualization import Visualizer

visualizer = Visualizer()
visualizer.generate_all_visualizations()
```

## 步骤 3: 查看结果

### 生成的文件

运行完成后，您将获得：

#### 数据文件
- `data/processed_data.csv` - 处理后的数据

#### 图表文件（results/figures/）
- `distribution_analysis.png` - 分布特征分析
- `correlation_matrix.png` - 相关性热图
- `score_trend_by_week.png` - 各周评分趋势
- `linear_regression_results.png` - 线性回归结果
- `anova_industry_scores.png` - 行业分析
- `ridge_regression_alpha.png` - 岭回归参数优化
- `rf_feature_importance.png` - 特征重要性（随机森林）
- `model_comparison.png` - 模型性能对比
- `cross_validation_results.png` - 交叉验证结果
- `residual_analysis.png` - 残差分析
- `learning_curve.png` - 学习曲线
- `score_evolution_top_10.png` - Top10选手评分演变
- `age_vs_performance.png` - 年龄与表现关系
- `industry_analysis.png` - 行业分析
- `season_trends.png` - 赛季趋势
- `score_distribution_by_week.png` - 各周评分分布
- `interactive_dashboard.html` - 交互式仪表板

#### 文档（paper/）
- `data_analysis_summary.md` - 数据分析总结
- `statistical_insights.md` - 统计洞察与建模总结

## 步骤 4: 使用结果撰写论文

### 论文结构建议

```
1. Introduction
   - 问题背景
   - 研究目标
   
2. Data Description
   - 使用: data_analysis_summary.md 第1节
   - 图表: distribution_analysis.png
   
3. Exploratory Data Analysis
   - 使用: data_analysis_summary.md 第2节
   - 图表: correlation_matrix.png, score_trend_by_week.png
   
4. Feature Engineering
   - 使用: data_analysis_summary.md 第3节
   - 图表: rf_feature_importance.png
   
5. Methodology
   - 使用: statistical_insights.md 第1-2节
   - 图表: model_comparison.png
   
6. Results
   - 使用: statistical_insights.md 第3节
   - 图表: linear_regression_results.png, anova_industry_scores.png
   
7. Model Validation
   - 使用: statistical_insights.md 第4节
   - 图表: cross_validation_results.png, residual_analysis.png
   
8. Discussion & Conclusion
   - 使用: data_analysis_summary.md 第4节
```

## 常见问题

### Q1: 如何修改模型参数？

编辑对应的Python模块（如 `src/prediction_models.py`），修改模型配置部分。

### Q2: 如何添加新特征？

在 `src/data_preprocessing.py` 的 `extract_score_features` 方法中添加。

### Q3: 如何自定义可视化？

修改 `src/visualization.py`，添加新的绘图函数。

### Q4: 图表显示中文乱码怎么办？

在代码开头添加：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # Mac/Linux/Windows
```

### Q5: 内存不足怎么办？

减少模型复杂度：
- 降低 `n_estimators`
- 减小 `max_depth`
- 使用更小的交叉验证折数

## 进阶使用

### 超参数优化

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# 网格搜索
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_}")
```

### 特征选择

```python
from sklearn.feature_selection import SelectKBest, f_regression

# 选择K个最佳特征
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)

# 查看选中的特征
selected_features = X.columns[selector.get_support()]
print(f"选中的特征: {selected_features}")
```

### SHAP值分析（模型解释）

```python
import shap

# 训练模型
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 计算SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 可视化
shap.summary_plot(shap_values, X_test)
```

## 获取帮助

如果遇到问题：
1. 查看代码中的文档字符串
2. 阅读 `paper/` 目录下的文档
3. 检查错误信息和堆栈跟踪
4. 确保所有依赖包已正确安装

## 祝你在MCM竞赛中取得好成绩！🎉
