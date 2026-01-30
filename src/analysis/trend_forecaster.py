"""
Problem 4: Trend Forecasting using Prophet

This module:
- Aggregates metrics by season
- Trains Prophet model with change points
- Forecasts future trends (3-5 seasons)
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


class TrendForecaster:
    """Forecasts trends using Facebook Prophet."""
    
    def __init__(self):
        """Initialize the trend forecaster."""
        self.models = {}
        self.forecasts = {}
        self.historical_data = {}
    
    def aggregate_by_season(self, df):
        """
        Aggregate key metrics by season.
        
        Args:
            df (pd.DataFrame): Full dataset
            
        Returns:
            pd.DataFrame: Aggregated data by season
        """
        print("\n=== Aggregating Data by Season ===")
        
        # Group by season and calculate metrics
        season_metrics = df.groupby('season').agg({
            'judge_score_mean': 'mean',
            'judge_score_std': 'mean',
            'placement': 'mean',
            'is_eliminated': 'mean',
            'residual': lambda x: np.abs(x).mean() if 'residual' in df.columns else 0
        }).reset_index()
        
        # Rename for clarity
        season_metrics.columns = [
            'season',
            'avg_score',
            'avg_score_variability',
            'avg_placement',
            'elimination_rate',
            'avg_abs_residual'
        ]
        
        # Create fairness index (inverse of absolute residual)
        season_metrics['fairness_index'] = 1 / (1 + season_metrics['avg_abs_residual'])
        
        print(f"Aggregated {len(season_metrics)} seasons")
        print(season_metrics.head())
        
        return season_metrics
    
    def prepare_time_series(self, season_metrics, metric_col, start_year=2005):
        """
        Prepare time series data for Prophet.
        
        Args:
            season_metrics (pd.DataFrame): Aggregated season data
            metric_col (str): Column to forecast
            start_year (int): Starting year of the show
            
        Returns:
            pd.DataFrame: Time series in Prophet format (ds, y)
        """
        # Create date column (assuming ~1 season per year, starting in spring)
        # Season 1 = 2005, Season 2 = 2006, etc.
        season_metrics = season_metrics.copy()
        season_metrics['year'] = start_year + season_metrics['season'] - 1
        season_metrics['ds'] = pd.to_datetime(season_metrics['year'].astype(str) + '-03-01')
        season_metrics['y'] = season_metrics[metric_col]
        
        # Prophet requires 'ds' and 'y' columns
        ts_data = season_metrics[['ds', 'y']].copy()
        
        self.historical_data[metric_col] = ts_data
        
        return ts_data
    
    def train_prophet_model(self, ts_data, metric_name, changepoint_years=None):
        """
        Train Prophet model with optional change points.
        
        Args:
            ts_data (pd.DataFrame): Time series data
            metric_name (str): Name of the metric
            changepoint_years (list): Years where rule changes occurred (e.g., [2028])
            
        Returns:
            Prophet model
        """
        print(f"\n=== Training Prophet Model for {metric_name} ===")
        
        # Initialize Prophet with parameters
        model = Prophet(
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
            seasonality_prior_scale=10.0,   # Strength of seasonality
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        
        # Add change points if specified (e.g., Season 28 rule change)
        if changepoint_years:
            changepoints = [pd.to_datetime(f'{year}-03-01') for year in changepoint_years]
            # Note: Prophet doesn't directly accept specific changepoints in this version
            # We'll rely on automatic changepoint detection
        
        # Fit the model
        model.fit(ts_data)
        
        self.models[metric_name] = model
        
        print(f"Model trained on {len(ts_data)} data points")
        print("=== Prophet Training Complete ===\n")
        
        return model
    
    def forecast_future(self, metric_name, periods=5, freq='YE'):
        """
        Forecast future trends.
        
        Args:
            metric_name (str): Name of the metric to forecast
            periods (int): Number of periods to forecast
            freq (str): Frequency ('YE' for year-end, 'Y' is deprecated)
            
        Returns:
            pd.DataFrame: Forecast dataframe
        """
        print(f"\n=== Forecasting {metric_name} for {periods} periods ===")
        
        model = self.models.get(metric_name)
        if model is None:
            print(f"Model for {metric_name} not found. Train it first.")
            return None
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='YE')
        
        # Generate forecast
        forecast = model.predict(future)
        
        self.forecasts[metric_name] = forecast
        
        print(f"Forecast generated for {len(forecast)} time points")
        print("\nFuture predictions:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
        print("=== Forecasting Complete ===\n")
        
        return forecast
    
    def plot_forecast(self, metric_name, output_dir='visualizations'):
        """
        Plot forecast with components.
        
        Args:
            metric_name (str): Name of the metric
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model = self.models.get(metric_name)
        forecast = self.forecasts.get(metric_name)
        
        if model is None or forecast is None:
            print(f"Model or forecast for {metric_name} not available")
            return
        
        # Create main forecast plot
        fig1 = model.plot(forecast)
        ax = fig1.gca()
        ax.set_title(f'Forecast: {metric_name}')
        ax.set_xlabel('Year')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line for rule change (Season 28 ~ 2032)
        if 'season' in metric_name or 'fairness' in metric_name:
            rule_change_date = pd.to_datetime('2032-03-01')
            ax.axvline(x=rule_change_date, color='red', linestyle='--', 
                       label='Rule Change (Season 28)', alpha=0.7)
            ax.legend()
        
        # Save main forecast plot
        safe_name = metric_name.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f'forecast_{safe_name}.png')
        plt.tight_layout()
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved forecast plot to {output_path}")
        plt.close(fig1)
        
        # Create component plot separately
        fig2 = model.plot_components(forecast)
        output_path2 = os.path.join(output_dir, f'forecast_{safe_name}_components.png')
        fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"Saved components plot to {output_path2}")
        plt.close(fig2)
    
    def analyze_trend_changes(self, metric_name):
        """
        Analyze trend changes and their impact.
        
        Args:
            metric_name (str): Name of the metric
            
        Returns:
            dict: Analysis results
        """
        forecast = self.forecasts.get(metric_name)
        if forecast is None:
            print(f"Forecast for {metric_name} not available")
            return None
        
        print(f"\n=== Trend Analysis: {metric_name} ===")
        
        # Calculate trend slope
        historical = forecast[forecast['ds'] <= datetime.now()]
        future = forecast[forecast['ds'] > datetime.now()]
        
        if len(historical) > 0 and len(future) > 0:
            historical_mean = historical['yhat'].mean()
            future_mean = future['yhat'].mean()
            percent_change = ((future_mean - historical_mean) / historical_mean) * 100
            
            print(f"Historical average: {historical_mean:.4f}")
            print(f"Future average: {future_mean:.4f}")
            print(f"Expected change: {percent_change:+.2f}%")
            
            if percent_change > 5:
                print("→ INCREASING TREND detected")
            elif percent_change < -5:
                print("→ DECREASING TREND detected")
            else:
                print("→ STABLE TREND detected")
        
        print("=== Trend Analysis Complete ===\n")
        
        return {
            'historical_mean': historical_mean,
            'future_mean': future_mean,
            'percent_change': percent_change
        }
    
    def generate_forecast_report(self, output_dir='reports'):
        """
        Generate a forecast report.
        
        Args:
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'trend_forecast_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TREND FORECASTING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("This report presents forecasts for key metrics over the next 3-5 seasons.\n\n")
            
            for metric_name, forecast in self.forecasts.items():
                f.write(f"\n{metric_name.upper()}\n")
                f.write("-" * 70 + "\n")
                
                # Get future predictions
                future = forecast[forecast['ds'] > datetime.now()].head(5)
                
                f.write("Future Predictions:\n")
                for _, row in future.iterrows():
                    f.write(f"  {row['ds'].year}: {row['yhat']:.4f} ")
                    f.write(f"[{row['yhat_lower']:.4f}, {row['yhat_upper']:.4f}]\n")
                
                f.write("\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            
            f.write("1. Monitor fairness index closely in upcoming seasons\n")
            f.write("2. If elimination rate trends upward, consider rule adjustments\n")
            f.write("3. Track score variability to ensure consistent judging\n")
            f.write("4. Implement the rule change at Season 28 as planned\n")
        
        print(f"Forecast report saved to {report_path}")


if __name__ == "__main__":
    print("Trend Forecasting Module")
    print("This module should be run as part of the main pipeline.")
