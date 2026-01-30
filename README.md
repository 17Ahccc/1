# MCM 2026 Problem C: Dancing with the Stars Analysis

## 项目概述 (Project Overview)

本项目为2026年美国大学生数学建模竞赛（MCM）C题解决方案，针对"Dancing with the Stars"（与星共舞）电视节目的数据进行深度统计分析与预测建模。

### 核心分析目标
1. **预测建模**: 基于选手特征和评分数据预测比赛结果
2. **影响因子分析**: 识别并量化影响比赛表现的关键因素
3. **数据可视化**: 挖掘数据特征并进行多维度可视化展示
4. **统计推断**: 使用严谨的统计方法支撑分析结论

## 项目结构 (Project Structure)

```
.
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python依赖包
├── data/
│   └── 2026_MCM_Problem_C_Data.csv    # 原始数据
├── src/
│   ├── data_preprocessing.py          # 数据预处理模块
│   ├── exploratory_analysis.py        # 探索性数据分析
│   ├── feature_engineering.py         # 特征工程
│   ├── statistical_models.py          # 统计模型实现
│   ├── prediction_models.py           # 预测模型（ML）
│   ├── visualization.py               # 数据可视化
│   └── model_evaluation.py            # 模型评估与验证
├── analysis/
│   ├── eda_report.ipynb              # EDA分析报告
│   ├── statistical_analysis.ipynb    # 统计分析报告
│   └── model_results.ipynb           # 模型结果展示
├── results/
│   ├── figures/                       # 生成的图表
│   ├── tables/                        # 统计表格
│   └── models/                        # 训练好的模型
└── paper/
    ├── data_analysis_summary.md       # 数据分析总结
    └── statistical_insights.md        # 统计洞察总结
```

## 快速开始 (Quick Start)

### 1. 环境配置
```bash
pip install -r requirements.txt
```

### 2. 运行完整分析流程
```bash
# 数据预处理
python src/data_preprocessing.py

# 探索性数据分析
python src/exploratory_analysis.py

# 特征工程
python src/feature_engineering.py

# 统计建模
python src/statistical_models.py

# 预测建模
python src/prediction_models.py

# 模型评估
python src/model_evaluation.py
```

### 3. 生成可视化结果
```bash
python src/visualization.py
```

## 核心功能模块 (Core Modules)

### 1. 数据预处理 (`data_preprocessing.py`)
- 数据清洗与缺失值处理
- 数据类型转换与编码
- 异常值检测与处理
- 数据标准化与归一化

### 2. 探索性数据分析 (`exploratory_analysis.py`)
- 描述性统计分析
- 分布特征分析
- 相关性分析
- 假设检验

### 3. 特征工程 (`feature_engineering.py`)
- 评分特征提取（均值、趋势、变异系数）
- 时序特征构建
- 交互特征生成
- 特征重要性评估

### 4. 统计模型 (`statistical_models.py`)
- 回归分析（线性、多项式、岭回归）
- 方差分析（ANOVA）
- 时间序列分析
- 生存分析（淘汰概率）

### 5. 预测模型 (`prediction_models.py`)
- 随机森林
- 梯度提升树（XGBoost、LightGBM）
- 神经网络
- 集成学习

### 6. 模型评估 (`model_evaluation.py`)
- 交叉验证
- 性能指标计算（MAE, RMSE, R²）
- 混淆矩阵与分类报告
- 模型泛化能力分析

### 7. 数据可视化 (`visualization.py`)
- 评分趋势图
- 特征重要性图
- 相关性热图
- 预测结果对比图
- 交互式可视化

## 美赛评分要点 (MCM Scoring Points)

### 1. 模型创新性
- ✅ 多模型融合策略
- ✅ 时序特征建模
- ✅ 影响因子量化分析
- ✅ 预测精度优化

### 2. 分析严谨性
- ✅ 统计假设检验
- ✅ 模型假设验证
- ✅ 残差分析
- ✅ 置信区间估计

### 3. 结果可解释性
- ✅ 特征重要性排序
- ✅ SHAP值分析
- ✅ 因果推断
- ✅ 敏感性分析

### 4. 泛化能力
- ✅ K折交叉验证
- ✅ 留一法验证
- ✅ 时间序列分割
- ✅ 鲁棒性测试

## 数据说明 (Data Description)

### 数据集特征
- **样本量**: 421条记录（422行包含表头）
- **时间跨度**: 多个赛季数据
- **特征维度**: 52个字段

### 主要字段
- `celebrity_name`: 选手姓名
- `celebrity_industry`: 选手行业背景
- `celebrity_age_during_season`: 参赛年龄
- `season`: 赛季编号
- `results`: 比赛结果
- `placement`: 最终排名
- `week*_judge*_score`: 每周各评委评分

## 技术栈 (Tech Stack)

- **Python 3.8+**
- **数据处理**: pandas, numpy
- **统计分析**: scipy, statsmodels
- **机器学习**: scikit-learn, xgboost, lightgbm
- **深度学习**: tensorflow/keras
- **可视化**: matplotlib, seaborn, plotly
- **报告生成**: jupyter notebook

## 预期产出 (Expected Outputs)

1. **数据分析报告**: 全面的EDA分析结果
2. **统计模型**: 回归分析、ANOVA等统计模型结果
3. **预测模型**: 高精度的排名预测模型
4. **可视化图表**: 20+张高质量图表用于论文
5. **特征洞察**: 影响比赛结果的关键因素分析
6. **模型评估**: 完整的模型性能评估报告
7. **论文素材**: 适配MCM论文结构的分析素材

## 获奖目标 (Award Goals)

- **目标奖项**: M奖（Meritorious Winner，一等奖）
- **核心竞争力**: 
  - 统计方法的创新性和严谨性
  - 预测模型的高准确率
  - 分析逻辑的完整性和可解释性
  - 可视化的专业性和洞察力

## 作者信息 (Authors)

MCM 2026 参赛团队
- 团队目标: 冲击M奖/H奖
- 擅长工具: Python
- 分析重点: 统计建模 + 预测精度 + 可视化
