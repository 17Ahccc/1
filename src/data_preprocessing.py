"""
数据预处理模块 (Data Preprocessing Module)
用于加载、清洗和预处理Dancing with the Stars数据集

功能:
1. 数据加载与初步探索
2. 缺失值处理
3. 数据类型转换
4. 特征编码
5. 数据标准化
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class DWTSDataPreprocessor:
    """Dancing with the Stars数据预处理器"""
    
    def __init__(self, data_path: str = 'data/2026_MCM_Problem_C_Data.csv'):
        """
        初始化预处理器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.feature_mappings = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据
        
        Returns:
            原始数据DataFrame
        """
        print("=" * 60)
        print("加载数据...")
        self.raw_data = pd.read_csv(self.data_path)
        print(f"数据形状: {self.raw_data.shape}")
        print(f"总样本数: {len(self.raw_data)}")
        print(f"总特征数: {len(self.raw_data.columns)}")
        print("=" * 60)
        return self.raw_data
    
    def explore_data_structure(self):
        """探索数据结构"""
        if self.raw_data is None:
            self.load_data()
        
        print("\n数据结构概览:")
        print("-" * 60)
        print(f"列名: {list(self.raw_data.columns)[:10]}...")
        print(f"\n前5行数据:")
        print(self.raw_data.head())
        print(f"\n数据类型:")
        print(self.raw_data.dtypes)
        print(f"\n基本统计信息:")
        print(self.raw_data.describe())
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        策略:
        1. 评分数据: N/A和0表示该周未参赛，保留为NaN
        2. 基本信息: 填充合理默认值或删除
        
        Args:
            df: 输入DataFrame
            
        Returns:
            处理后的DataFrame
        """
        print("\n处理缺失值...")
        df = df.copy()
        
        # 统计缺失值
        missing_info = df.isnull().sum()
        print(f"缺失值统计:")
        print(missing_info[missing_info > 0])
        
        # 处理评分列: 将'N/A'字符串转换为NaN
        score_columns = [col for col in df.columns if 'score' in col.lower()]
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 0分表示未参赛，转换为NaN
        df[score_columns] = df[score_columns].replace(0, np.nan)
        
        print(f"评分列缺失值处理完成")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        编码分类特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            编码后的DataFrame
        """
        print("\n编码分类特征...")
        df = df.copy()
        
        # 编码行业类别
        if 'celebrity_industry' in df.columns:
            industry_mapping = {
                industry: idx 
                for idx, industry in enumerate(df['celebrity_industry'].unique())
            }
            self.feature_mappings['celebrity_industry'] = industry_mapping
            df['celebrity_industry_encoded'] = df['celebrity_industry'].map(industry_mapping)
        
        # 编码结果类别
        if 'results' in df.columns:
            df['is_winner'] = df['results'].str.contains('1st Place', na=False).astype(int)
            df['is_eliminated'] = df['results'].str.contains('Eliminated', na=False).astype(int)
            
            # 提取淘汰周数
            df['eliminated_week'] = df['results'].str.extract(r'Week (\d+)', expand=False)
            df['eliminated_week'] = pd.to_numeric(df['eliminated_week'], errors='coerce')
        
        # 编码国家/地区
        if 'celebrity_homecountry/region' in df.columns:
            country_mapping = {
                country: idx 
                for idx, country in enumerate(df['celebrity_homecountry/region'].unique())
            }
            self.feature_mappings['celebrity_homecountry/region'] = country_mapping
            df['celebrity_homecountry_encoded'] = df['celebrity_homecountry/region'].map(country_mapping)
        
        print("分类特征编码完成")
        return df
    
    def extract_score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取评分特征
        
        为每个选手计算:
        1. 平均分
        2. 最高分
        3. 最低分
        4. 分数标准差
        5. 分数变异系数
        6. 分数趋势（线性回归斜率）
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加评分特征后的DataFrame
        """
        print("\n提取评分特征...")
        df = df.copy()
        
        # 获取所有评分列
        score_columns = [col for col in df.columns if 'score' in col.lower()]
        
        # 为每行计算统计特征
        df['avg_score'] = df[score_columns].mean(axis=1, skipna=True)
        df['max_score'] = df[score_columns].max(axis=1, skipna=True)
        df['min_score'] = df[score_columns].min(axis=1, skipna=True)
        df['std_score'] = df[score_columns].std(axis=1, skipna=True)
        df['cv_score'] = df['std_score'] / df['avg_score']  # 变异系数
        
        # 计算有效评分周数
        df['num_weeks_participated'] = df[score_columns].notna().sum(axis=1)
        
        # 计算分数趋势（简化版：最后一周减第一周）
        first_week_scores = df[[col for col in score_columns if 'week1' in col]].mean(axis=1)
        last_valid_scores = df[score_columns].iloc[:, -4:].mean(axis=1, skipna=True)
        df['score_trend'] = last_valid_scores - first_week_scores
        
        print("评分特征提取完成")
        return df
    
    def process(self) -> pd.DataFrame:
        """
        执行完整的数据预处理流程
        
        Returns:
            处理后的数据
        """
        print("\n" + "=" * 60)
        print("开始数据预处理流程")
        print("=" * 60)
        
        # 加载数据
        if self.raw_data is None:
            self.load_data()
        
        # 处理步骤
        df = self.raw_data.copy()
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df)
        df = self.extract_score_features(df)
        
        self.processed_data = df
        
        print("\n" + "=" * 60)
        print("数据预处理完成!")
        print(f"处理后数据形状: {df.shape}")
        print(f"新增特征数: {len(df.columns) - len(self.raw_data.columns)}")
        print("=" * 60)
        
        return df
    
    def save_processed_data(self, output_path: str = 'data/processed_data.csv'):
        """
        保存处理后的数据
        
        Args:
            output_path: 输出文件路径
        """
        if self.processed_data is None:
            self.process()
        
        self.processed_data.to_csv(output_path, index=False)
        print(f"\n处理后数据已保存至: {output_path}")
    
    def get_summary_statistics(self) -> Dict:
        """
        获取数据摘要统计
        
        Returns:
            统计信息字典
        """
        if self.processed_data is None:
            self.process()
        
        stats = {
            'total_samples': len(self.processed_data),
            'total_features': len(self.processed_data.columns),
            'total_seasons': self.processed_data['season'].nunique(),
            'total_celebrities': self.processed_data['celebrity_name'].nunique(),
            'industries': self.processed_data['celebrity_industry'].value_counts().to_dict(),
            'avg_age': self.processed_data['celebrity_age_during_season'].mean(),
            'avg_score': self.processed_data['avg_score'].mean(),
        }
        
        return stats


def main():
    """主函数"""
    # 创建预处理器实例
    preprocessor = DWTSDataPreprocessor()
    
    # 探索原始数据
    preprocessor.load_data()
    preprocessor.explore_data_structure()
    
    # 执行预处理
    processed_data = preprocessor.process()
    
    # 保存处理后的数据
    preprocessor.save_processed_data()
    
    # 获取并打印摘要统计
    stats = preprocessor.get_summary_statistics()
    print("\n" + "=" * 60)
    print("数据摘要统计:")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return processed_data


if __name__ == "__main__":
    processed_data = main()
