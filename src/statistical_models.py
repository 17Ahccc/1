"""
统计模型模块 (Statistical Models Module)
使用严谨的统计方法分析Dancing with the Stars数据

模型包括:
1. 多元线性回归
2. 逻辑回归（淘汰预测）
3. 生存分析
4. 方差分析(ANOVA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')


class StatisticalModeler:
    """统计建模器"""
    
    def __init__(self, data_path: str = 'data/processed_data.csv'):
        """
        初始化建模器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.results = {}
        self.results_dir = 'results/figures'
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成: {self.data.shape}")
        return self.data
    
    def multiple_linear_regression(self):
        """
        多元线性回归分析
        预测目标: 最终排名(placement)
        """
        print("\n" + "=" * 60)
        print("多元线性回归分析")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 选择特征和目标
        feature_cols = [
            'celebrity_age_during_season',
            'avg_score',
            'std_score',
            'num_weeks_participated',
            'score_trend'
        ]
        
        # 过滤有效数据
        available_cols = [col for col in feature_cols if col in self.data.columns]
        df = self.data[available_cols + ['placement']].dropna()
        
        X = df[available_cols]
        y = df['placement']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # 评估
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\n模型性能:")
        print(f"  训练集 R²: {train_r2:.4f}")
        print(f"  测试集 R²: {test_r2:.4f}")
        print(f"  训练集 RMSE: {train_rmse:.4f}")
        print(f"  测试集 RMSE: {test_rmse:.4f}")
        
        # 特征重要性
        print(f"\n特征系数:")
        for i, col in enumerate(available_cols):
            print(f"  {col}: {model.coef_[i]:.4f}")
        
        # 使用statsmodels进行详细统计检验
        X_train_sm = sm.add_constant(X_train_scaled)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        print(f"\n详细统计摘要:")
        print(model_sm.summary())
        
        # 保存模型
        self.models['linear_regression'] = {
            'model': model,
            'scaler': scaler,
            'features': available_cols
        }
        
        self.results['linear_regression'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'coefficients': dict(zip(available_cols, model.coef_))
        }
        
        # 可视化：实际值 vs 预测值
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 训练集
        axes[0].scatter(y_train, y_pred_train, alpha=0.5, color='blue')
        axes[0].plot([y_train.min(), y_train.max()], 
                    [y_train.min(), y_train.max()], 
                    'r--', linewidth=2)
        axes[0].set_xlabel('Actual Placement')
        axes[0].set_ylabel('Predicted Placement')
        axes[0].set_title(f'Training Set (R² = {train_r2:.3f})')
        axes[0].grid(True, alpha=0.3)
        
        # 测试集
        axes[1].scatter(y_test, y_pred_test, alpha=0.5, color='green')
        axes[1].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', linewidth=2)
        axes[1].set_xlabel('Actual Placement')
        axes[1].set_ylabel('Predicted Placement')
        axes[1].set_title(f'Test Set (R² = {test_r2:.3f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/linear_regression_results.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n回归结果图已保存至: {self.results_dir}/linear_regression_results.png")
        plt.close()
    
    def logistic_regression_elimination(self):
        """
        逻辑回归分析
        预测目标: 是否被淘汰
        """
        print("\n" + "=" * 60)
        print("逻辑回归分析 - 淘汰预测")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 准备数据
        if 'is_eliminated' not in self.data.columns:
            print("缺少淘汰标签，跳过此分析")
            return
        
        feature_cols = [
            'celebrity_age_during_season',
            'avg_score',
            'std_score',
            'num_weeks_participated'
        ]
        
        available_cols = [col for col in feature_cols if col in self.data.columns]
        df = self.data[available_cols + ['is_eliminated']].dropna()
        
        X = df[available_cols]
        y = df['is_eliminated']
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # 评估
        train_acc = (y_train == y_pred_train).mean()
        test_acc = (y_test == y_pred_test).mean()
        
        print(f"\n模型性能:")
        print(f"  训练集准确率: {train_acc:.4f}")
        print(f"  测试集准确率: {test_acc:.4f}")
        
        print(f"\n分类报告（测试集）:")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=['Not Eliminated', 'Eliminated']))
        
        # 特征系数
        print(f"\n特征系数（Odds Ratio）:")
        for i, col in enumerate(available_cols):
            odds_ratio = np.exp(model.coef_[0][i])
            print(f"  {col}: {odds_ratio:.4f}")
        
        # 保存模型
        self.models['logistic_regression'] = {
            'model': model,
            'scaler': scaler,
            'features': available_cols
        }
        
        self.results['logistic_regression'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'coefficients': dict(zip(available_cols, model.coef_[0]))
        }
    
    def anova_analysis(self):
        """
        方差分析（ANOVA）
        分析不同行业对平均分的影响
        """
        print("\n" + "=" * 60)
        print("方差分析（ANOVA）- 行业对分数的影响")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        if 'celebrity_industry' not in self.data.columns:
            print("缺少行业信息，跳过此分析")
            return
        
        # 选择主要行业
        top_industries = self.data['celebrity_industry'].value_counts().head(5).index
        df = self.data[self.data['celebrity_industry'].isin(top_industries)].copy()
        df = df[['celebrity_industry', 'avg_score']].dropna()
        
        # 执行ANOVA
        formula = 'avg_score ~ C(celebrity_industry)'
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        print(f"\nANOVA结果:")
        print(anova_table)
        
        # 解释
        p_value = anova_table['PR(>F)']['C(celebrity_industry)']
        if p_value < 0.05:
            print(f"\n结论: 不同行业的平均分存在显著差异 (p = {p_value:.4f} < 0.05)")
        else:
            print(f"\n结论: 不同行业的平均分无显著差异 (p = {p_value:.4f} >= 0.05)")
        
        # 可视化
        plt.figure(figsize=(12, 6))
        df.boxplot(column='avg_score', by='celebrity_industry', figsize=(12, 6))
        plt.xlabel('Industry')
        plt.ylabel('Average Score')
        plt.title('Average Score by Industry')
        plt.suptitle('')  # 移除自动标题
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/anova_industry_scores.png', 
                   dpi=300, bbox_inches='tight')
        print(f"ANOVA结果图已保存至: {self.results_dir}/anova_industry_scores.png")
        plt.close()
        
        self.results['anova'] = {
            'p_value': p_value,
            'f_statistic': anova_table['F']['C(celebrity_industry)']
        }
    
    def ridge_regression(self):
        """
        岭回归（Ridge Regression）
        用于处理多重共线性问题
        """
        print("\n" + "=" * 60)
        print("岭回归分析")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 选择特征
        feature_cols = [
            'celebrity_age_during_season',
            'avg_score',
            'max_score',
            'min_score',
            'std_score',
            'num_weeks_participated',
            'score_trend'
        ]
        
        available_cols = [col for col in feature_cols if col in self.data.columns]
        df = self.data[available_cols + ['placement']].dropna()
        
        X = df[available_cols]
        y = df['placement']
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 测试不同的alpha值
        alphas = [0.001, 0.01, 0.1, 1, 10, 100]
        train_scores = []
        test_scores = []
        
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train)
            train_scores.append(model.score(X_train_scaled, y_train))
            test_scores.append(model.score(X_test_scaled, y_test))
        
        # 找到最佳alpha
        best_idx = np.argmax(test_scores)
        best_alpha = alphas[best_idx]
        
        print(f"\n最佳正则化参数 α: {best_alpha}")
        print(f"对应的测试集 R²: {test_scores[best_idx]:.4f}")
        
        # 使用最佳alpha训练最终模型
        best_model = Ridge(alpha=best_alpha)
        best_model.fit(X_train_scaled, y_train)
        
        # 可视化alpha对性能的影响
        plt.figure(figsize=(10, 6))
        plt.semilogx(alphas, train_scores, 'o-', label='Training Score', linewidth=2)
        plt.semilogx(alphas, test_scores, 's-', label='Test Score', linewidth=2)
        plt.axvline(best_alpha, color='r', linestyle='--', label=f'Best α = {best_alpha}')
        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel('R² Score')
        plt.title('Ridge Regression: Model Performance vs Regularization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/ridge_regression_alpha.png', 
                   dpi=300, bbox_inches='tight')
        print(f"岭回归结果图已保存至: {self.results_dir}/ridge_regression_alpha.png")
        plt.close()
        
        self.models['ridge_regression'] = {
            'model': best_model,
            'scaler': scaler,
            'features': available_cols,
            'best_alpha': best_alpha
        }
        
        self.results['ridge_regression'] = {
            'best_alpha': best_alpha,
            'test_r2': test_scores[best_idx],
            'coefficients': dict(zip(available_cols, best_model.coef_))
        }
    
    def run_all_models(self):
        """运行所有统计模型"""
        print("\n" + "=" * 80)
        print("开始统计建模")
        print("=" * 80)
        
        # 加载数据
        self.load_data()
        
        # 运行各个模型
        self.multiple_linear_regression()
        self.logistic_regression_elimination()
        self.anova_analysis()
        self.ridge_regression()
        
        print("\n" + "=" * 80)
        print("统计建模完成!")
        print("=" * 80)
        
        # 打印汇总结果
        print("\n模型性能汇总:")
        print("-" * 60)
        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            for key, value in result.items():
                if not isinstance(value, dict):
                    print(f"  {key}: {value}")


def main():
    """主函数"""
    # 创建建模器实例
    modeler = StatisticalModeler()
    
    # 运行所有模型
    modeler.run_all_models()


if __name__ == "__main__":
    main()
