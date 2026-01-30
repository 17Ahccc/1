"""
预测模型模块 (Prediction Models Module)
使用机器学习方法进行高精度预测

模型包括:
1. 随机森林 (Random Forest)
2. 梯度提升树 (XGBoost, LightGBM)
3. 集成学习
4. 神经网络（可选）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class PredictionModeler:
    """预测建模器"""
    
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
    
    def prepare_features(self, target_col='placement'):
        """
        准备特征和目标变量
        
        Args:
            target_col: 目标变量列名
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # 选择特征
        feature_cols = [
            'celebrity_age_during_season',
            'season',
            'avg_score',
            'max_score',
            'min_score',
            'std_score',
            'cv_score',
            'num_weeks_participated',
            'score_trend'
        ]
        
        # 过滤可用特征
        available_cols = [col for col in feature_cols if col in self.data.columns]
        
        # 准备数据
        df = self.data[available_cols + [target_col]].dropna()
        
        X = df[available_cols]
        y = df[target_col]
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, available_cols
    
    def random_forest_model(self):
        """
        随机森林模型
        用于排名预测
        """
        print("\n" + "=" * 60)
        print("随机森林模型")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 准备数据
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_features()
        
        # 训练模型
        print("\n训练随机森林模型...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\n模型性能:")
        print(f"  训练集 R²: {train_r2:.4f}")
        print(f"  测试集 R²: {test_r2:.4f}")
        print(f"  训练集 RMSE: {train_rmse:.4f}")
        print(f"  测试集 RMSE: {test_rmse:.4f}")
        print(f"  测试集 MAE: {test_mae:.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n特征重要性:")
        print(feature_importance)
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Random Forest: Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/rf_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存")
        plt.close()
        
        # 保存模型和结果
        self.models['random_forest'] = {
            'model': model,
            'features': feature_cols
        }
        
        self.results['random_forest'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'feature_importance': feature_importance.to_dict()
        }
        
        return model, test_r2
    
    def xgboost_model(self):
        """
        XGBoost模型
        高性能梯度提升模型
        """
        print("\n" + "=" * 60)
        print("XGBoost模型")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 准备数据
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_features()
        
        # 训练模型
        print("\n训练XGBoost模型...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\n模型性能:")
        print(f"  训练集 R²: {train_r2:.4f}")
        print(f"  测试集 R²: {test_r2:.4f}")
        print(f"  训练集 RMSE: {train_rmse:.4f}")
        print(f"  测试集 RMSE: {test_rmse:.4f}")
        print(f"  测试集 MAE: {test_mae:.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n特征重要性:")
        print(feature_importance)
        
        # 保存模型和结果
        self.models['xgboost'] = {
            'model': model,
            'features': feature_cols
        }
        
        self.results['xgboost'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'feature_importance': feature_importance.to_dict()
        }
        
        return model, test_r2
    
    def lightgbm_model(self):
        """
        LightGBM模型
        快速高效的梯度提升模型
        """
        print("\n" + "=" * 60)
        print("LightGBM模型")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 准备数据
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_features()
        
        # 训练模型
        print("\n训练LightGBM模型...")
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\n模型性能:")
        print(f"  训练集 R²: {train_r2:.4f}")
        print(f"  测试集 R²: {test_r2:.4f}")
        print(f"  训练集 RMSE: {train_rmse:.4f}")
        print(f"  测试集 RMSE: {test_rmse:.4f}")
        print(f"  测试集 MAE: {test_mae:.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n特征重要性:")
        print(feature_importance)
        
        # 保存模型和结果
        self.models['lightgbm'] = {
            'model': model,
            'features': feature_cols
        }
        
        self.results['lightgbm'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'feature_importance': feature_importance.to_dict()
        }
        
        return model, test_r2
    
    def ensemble_model(self):
        """
        集成学习模型
        结合多个模型的预测结果
        """
        print("\n" + "=" * 60)
        print("集成学习模型")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 准备数据
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_features()
        
        # 训练各个基模型（如果还没训练）
        if 'random_forest' not in self.models:
            self.random_forest_model()
        if 'xgboost' not in self.models:
            self.xgboost_model()
        if 'lightgbm' not in self.models:
            self.lightgbm_model()
        
        # 获取各个模型的预测
        rf_pred = self.models['random_forest']['model'].predict(X_test)
        xgb_pred = self.models['xgboost']['model'].predict(X_test)
        lgb_pred = self.models['lightgbm']['model'].predict(X_test)
        
        # 简单平均集成
        ensemble_pred = (rf_pred + xgb_pred + lgb_pred) / 3
        
        # 评估
        test_r2 = r2_score(y_test, ensemble_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        test_mae = mean_absolute_error(y_test, ensemble_pred)
        
        print(f"\n集成模型性能:")
        print(f"  测试集 R²: {test_r2:.4f}")
        print(f"  测试集 RMSE: {test_rmse:.4f}")
        print(f"  测试集 MAE: {test_mae:.4f}")
        
        # 与各单独模型比较
        print(f"\n模型性能对比:")
        print(f"  Random Forest R²: {self.results['random_forest']['test_r2']:.4f}")
        print(f"  XGBoost R²: {self.results['xgboost']['test_r2']:.4f}")
        print(f"  LightGBM R²: {self.results['lightgbm']['test_r2']:.4f}")
        print(f"  Ensemble R²: {test_r2:.4f}")
        
        # 可视化模型比较
        models_names = ['RF', 'XGB', 'LGB', 'Ensemble']
        r2_scores = [
            self.results['random_forest']['test_r2'],
            self.results['xgboost']['test_r2'],
            self.results['lightgbm']['test_r2'],
            test_r2
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models_names, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison')
        plt.ylim([0, 1])
        for i, v in enumerate(r2_scores):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存")
        plt.close()
        
        self.results['ensemble'] = {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
    
    def run_all_models(self):
        """运行所有预测模型"""
        print("\n" + "=" * 80)
        print("开始预测建模")
        print("=" * 80)
        
        # 加载数据
        self.load_data()
        
        # 运行各个模型
        self.random_forest_model()
        self.xgboost_model()
        self.lightgbm_model()
        self.ensemble_model()
        
        print("\n" + "=" * 80)
        print("预测建模完成!")
        print("=" * 80)
        
        # 打印汇总结果
        print("\n模型性能汇总:")
        print("-" * 60)
        for model_name in ['random_forest', 'xgboost', 'lightgbm', 'ensemble']:
            if model_name in self.results:
                result = self.results[model_name]
                print(f"\n{model_name.upper()}:")
                print(f"  Test R²: {result.get('test_r2', 'N/A')}")
                print(f"  Test RMSE: {result.get('test_rmse', 'N/A')}")
                print(f"  Test MAE: {result.get('test_mae', 'N/A')}")


def main():
    """主函数"""
    # 创建建模器实例
    modeler = PredictionModeler()
    
    # 运行所有模型
    modeler.run_all_models()


if __name__ == "__main__":
    main()
