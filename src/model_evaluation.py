"""
模型评估模块 (Model Evaluation Module)
对模型进行全面的性能评估和泛化能力分析

包括:
1. 交叉验证
2. 性能指标计算
3. 残差分析
4. 泛化能力评估
5. 敏感性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, data_path: str = 'data/processed_data.csv'):
        """
        初始化评估器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.results_dir = 'results/figures'
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成: {self.data.shape}")
        return self.data
    
    def prepare_data(self, target_col='placement'):
        """准备特征和目标"""
        feature_cols = [
            'celebrity_age_during_season',
            'avg_score',
            'std_score',
            'num_weeks_participated',
            'score_trend'
        ]
        
        available_cols = [col for col in feature_cols if col in self.data.columns]
        df = self.data[available_cols + [target_col]].dropna()
        
        X = df[available_cols]
        y = df[target_col]
        
        return X, y, available_cols
    
    def cross_validation_analysis(self):
        """
        交叉验证分析
        使用K折交叉验证评估模型稳定性
        """
        print("\n" + "=" * 60)
        print("K折交叉验证分析")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        X, y, feature_cols = self.prepare_data()
        
        # 定义模型
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        # K折交叉验证 (K=5)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_results = {}
        
        for model_name, model in models.items():
            print(f"\n{model_name}:")
            
            # R² scores
            r2_scores = cross_val_score(model, X, y, cv=kfold, 
                                       scoring='r2', n_jobs=-1)
            
            # Negative MSE scores (转换为RMSE)
            neg_mse_scores = cross_val_score(model, X, y, cv=kfold, 
                                             scoring='neg_mean_squared_error', n_jobs=-1)
            rmse_scores = np.sqrt(-neg_mse_scores)
            
            # 统计结果
            print(f"  R² scores: {r2_scores}")
            print(f"  Mean R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
            print(f"  RMSE scores: {rmse_scores}")
            print(f"  Mean RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
            
            cv_results[model_name] = {
                'r2_scores': r2_scores,
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std(),
                'rmse_scores': rmse_scores,
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std()
            }
        
        # 可视化交叉验证结果
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # R² scores
        for i, (model_name, results) in enumerate(cv_results.items()):
            axes[0].bar(i, results['r2_mean'], yerr=results['r2_std'], 
                       capsize=10, alpha=0.7, label=model_name)
        
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Cross-Validation R² Scores')
        axes[0].set_xticks(range(len(cv_results)))
        axes[0].set_xticklabels(cv_results.keys())
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # RMSE scores
        for i, (model_name, results) in enumerate(cv_results.items()):
            axes[1].bar(i, results['rmse_mean'], yerr=results['rmse_std'], 
                       capsize=10, alpha=0.7, label=model_name, color='coral')
        
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Cross-Validation RMSE')
        axes[1].set_xticks(range(len(cv_results)))
        axes[1].set_xticklabels(cv_results.keys())
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/cross_validation_results.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n交叉验证结果图已保存")
        plt.close()
        
        return cv_results
    
    def residual_analysis(self):
        """
        残差分析
        检查模型预测误差的分布特征
        """
        print("\n" + "=" * 60)
        print("残差分析")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        X, y, feature_cols = self.prepare_data()
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        
        # 预测
        y_pred = model.predict(X)
        
        # 计算残差
        residuals = y - y_pred
        
        # 统计信息
        print(f"\n残差统计:")
        print(f"  均值: {residuals.mean():.4f}")
        print(f"  标准差: {residuals.std():.4f}")
        print(f"  最小值: {residuals.min():.4f}")
        print(f"  最大值: {residuals.max():.4f}")
        
        # 可视化残差
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Residual Analysis', fontsize=16, y=0.995)
        
        # 1. 残差 vs 预测值
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差分布直方图
        axes[0, 1].hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residual Distribution')
        
        # 3. Q-Q图（正态性检验）
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # 4. 残差的时间序列图
        axes[1, 1].plot(residuals, marker='o', linestyle='', alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Sequence Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/residual_analysis.png', 
                   dpi=300, bbox_inches='tight')
        print(f"残差分析图已保存")
        plt.close()
    
    def learning_curve_analysis(self):
        """
        学习曲线分析
        评估训练集大小对模型性能的影响
        """
        print("\n" + "=" * 60)
        print("学习曲线分析")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        X, y, feature_cols = self.prepare_data()
        
        # 定义训练集大小
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_scores_mean = []
        train_scores_std = []
        test_scores_mean = []
        test_scores_std = []
        
        for train_size in train_sizes:
            n_samples = int(len(X) * train_size)
            
            if n_samples < 10:
                continue
            
            # 使用5折交叉验证
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            train_scores = []
            test_scores = []
            
            for train_idx, test_idx in kfold.split(X):
                # 获取训练集的子集
                X_train = X.iloc[train_idx[:n_samples]]
                y_train = y.iloc[train_idx[:n_samples]]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]
                
                # 训练模型
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # 评估
                train_scores.append(model.score(X_train, y_train))
                test_scores.append(model.score(X_test, y_test))
            
            train_scores_mean.append(np.mean(train_scores))
            train_scores_std.append(np.std(train_scores))
            test_scores_mean.append(np.mean(test_scores))
            test_scores_std.append(np.std(test_scores))
        
        # 绘制学习曲线
        train_sizes_abs = [int(len(X) * s) for s in train_sizes[:len(train_scores_mean)]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue', 
                label='Training score', linewidth=2)
        plt.fill_between(train_sizes_abs, 
                        np.array(train_scores_mean) - np.array(train_scores_std),
                        np.array(train_scores_mean) + np.array(train_scores_std),
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes_abs, test_scores_mean, 's-', color='red', 
                label='Cross-validation score', linewidth=2)
        plt.fill_between(train_sizes_abs, 
                        np.array(test_scores_mean) - np.array(test_scores_std),
                        np.array(test_scores_mean) + np.array(test_scores_std),
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title('Learning Curve Analysis', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/learning_curve.png', 
                   dpi=300, bbox_inches='tight')
        print(f"学习曲线图已保存")
        plt.close()
    
    def run_full_evaluation(self):
        """运行完整评估流程"""
        print("\n" + "=" * 80)
        print("开始模型评估")
        print("=" * 80)
        
        # 加载数据
        self.load_data()
        
        # 执行各项评估
        self.cross_validation_analysis()
        self.residual_analysis()
        self.learning_curve_analysis()
        
        print("\n" + "=" * 80)
        print("模型评估完成!")
        print("=" * 80)


def main():
    """主函数"""
    # 创建评估器实例
    evaluator = ModelEvaluator()
    
    # 运行完整评估
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
