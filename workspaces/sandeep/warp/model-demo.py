import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import matplotlib.pyplot as plt
import os

class CostSavingsAnalyzer:
    """Class to analyze cost savings between actual and predicted prices"""
    
    def __init__(self, avg_daily_consumption: float = 100.0):
        """
        Initialize the analyzer
        
        Args:
            avg_daily_consumption: Average daily consumption in units (default: 100)
        """
        self.avg_daily_consumption = avg_daily_consumption
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_path = Path('workspaces/sandeep/warp/logs')
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / 'cost_savings.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('CostSavingsAnalyzer')
    
    def load_rolling_predictions(self, file_path: str) -> pd.DataFrame:
        """
        Load and prepare data from rolling predictions CSV
        
        Args:
            file_path: Path to the rolling predictions CSV file
        
        Returns:
            DataFrame with actual and predicted prices
        """
        try:
            self.logger.info(f"Loading rolling predictions from {file_path}")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Log the column names to help debug
            self.logger.info(f"Available columns: {df.columns.tolist()}")
            
            # First, let's see what the data looks like
            self.logger.info("\nFirst few rows of the data:")
            self.logger.info(df.head().to_string())
            
            # Rename columns based on the actual CSV structure
            column_mapping = {
                'Timestamp': 'timestamp',
                'Actual': 'actual_price',
                'Predicted_1d_ahead': 'predicted_price'
            }
            
            # Check which columns exist and create mapping
            existing_columns = {}
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    existing_columns[old_col] = new_col
                else:
                    self.logger.warning(f"Column {old_col} not found in CSV")
            
            # Rename existing columns
            df = df.rename(columns=existing_columns)
            
            # Ensure timestamp column is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Select and reorder columns
            required_columns = ['timestamp', 'actual_price', 'predicted_price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            df = df[required_columns]
            
            self.logger.info(f"Loaded data with shape: {df.shape}")
            self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading rolling predictions: {str(e)}")
            self.logger.error(f"Available columns: {df.columns.tolist() if 'df' in locals() else 'No DataFrame loaded'}")
            raise
    
    def optimize_consumption(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize consumption based on predicted prices"""
        self.logger.info("Starting consumption optimization...")
        
        # Create a copy of the dataframe
        df = df.copy()
        
        # Use actual prices when predicted prices are NaN
        df['price_for_optimization'] = df['predicted_price'].fillna(df['actual_price'])
        
        # Calculate price percentiles
        price_percentiles = df['price_for_optimization'].quantile([0.33, 0.66])
        low_price_threshold = price_percentiles[0.33]
        high_price_threshold = price_percentiles[0.66]
        
        # Initialize optimized consumption
        df['optimized_consumption'] = self.avg_daily_consumption
        
        # Count days in each price category
        low_price_days = (df['price_for_optimization'] <= low_price_threshold).sum()
        high_price_days = (df['price_for_optimization'] >= high_price_threshold).sum()
        medium_price_days = len(df) - low_price_days - high_price_days
        
        # Calculate consumption adjustments
        if low_price_days > 0:
            # On low price days, consume 20% more than average
            low_price_consumption = self.avg_daily_consumption * 1.2
            df.loc[df['price_for_optimization'] <= low_price_threshold, 'optimized_consumption'] = low_price_consumption
        
        if high_price_days > 0:
            # On high price days, consume 20% less than average
            high_price_consumption = self.avg_daily_consumption * 0.8
            df.loc[df['price_for_optimization'] >= high_price_threshold, 'optimized_consumption'] = high_price_consumption
        
        # Calculate total consumption
        total_consumption = df['optimized_consumption'].sum()
        target_consumption = self.avg_daily_consumption * len(df)
        
        # Adjust consumption to match target if needed
        if total_consumption != target_consumption:
            adjustment_factor = target_consumption / total_consumption
            df['optimized_consumption'] *= adjustment_factor
        
        # Log optimization summary
        self.logger.info("\nOptimization Summary:")
        self.logger.info(f"Low price days (<= {low_price_threshold:.4f}): {low_price_days} days")
        self.logger.info(f"High price days (>= {high_price_threshold:.4f}): {high_price_days} days")
        self.logger.info(f"Medium price days: {medium_price_days} days")
        
        return df
    
    def calculate_cost_savings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cost savings between actual and predicted prices"""
        self.logger.info("Starting cost savings calculation...")
        
        # Validate input DataFrame
        required_columns = ['actual_price', 'predicted_price']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input DataFrame must contain columns: {required_columns}")
        
        # Create a copy to avoid modifying the input
        results_df = df.copy()
        
        # Calculate costs with average consumption
        results_df['avg_consumption_cost'] = results_df['actual_price'] * self.avg_daily_consumption
        
        # Optimize consumption based on predicted prices
        results_df = self.optimize_consumption(results_df)
        
        # Calculate costs with optimized consumption
        results_df['optimized_cost'] = results_df['actual_price'] * results_df['optimized_consumption']
        
        # Calculate savings
        results_df['savings'] = results_df['avg_consumption_cost'] - results_df['optimized_cost']
        results_df['cumulative_savings'] = results_df['savings'].cumsum()
        
        # Calculate total consumption
        results_df['total_consumption'] = results_df['optimized_consumption'].sum()
        results_df['avg_consumption'] = results_df['optimized_consumption'].mean()
        
        # Log detailed analysis
        self.logger.info("\nDetailed Analysis:")
        self.logger.info("\nPrice Statistics:")
        self.logger.info(f"Mean actual price: {results_df['actual_price'].mean():.2f}")
        self.logger.info(f"Min actual price: {results_df['actual_price'].min():.2f}")
        self.logger.info(f"Max actual price: {results_df['actual_price'].max():.2f}")
        
        self.logger.info("\nConsumption Statistics:")
        self.logger.info(f"Mean optimized consumption: {results_df['optimized_consumption'].mean():.2f}")
        self.logger.info(f"Min optimized consumption: {results_df['optimized_consumption'].min():.2f}")
        self.logger.info(f"Max optimized consumption: {results_df['optimized_consumption'].max():.2f}")
        
        self.logger.info("\nCost Statistics:")
        self.logger.info(f"Total average consumption cost: {results_df['avg_consumption_cost'].sum():.2f}")
        self.logger.info(f"Total optimized cost: {results_df['optimized_cost'].sum():.2f}")
        self.logger.info(f"Total savings: {results_df['savings'].sum():.2f}")
        
        # Log daily details for first few days
        self.logger.info("\nDaily Details (first 5 days):")
        daily_details = results_df[['timestamp', 'actual_price', 'predicted_price', 
                                  'optimized_consumption', 'avg_consumption_cost', 
                                  'optimized_cost', 'savings']].head()
        self.logger.info("\n" + daily_details.to_string())
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate and save the plot
        plot_path = os.path.join(plots_dir, 'cost_savings_analysis.png')
        self.plot_savings(results_df, save_path=plot_path)
        
        self.logger.info("Cost savings calculation completed")
        return results_df
    
    def analyze_savings(self, df: pd.DataFrame) -> Dict:
        """Analyze the savings and return key metrics"""
        self.logger.info("Starting savings analysis...")
        
        total_avg_cost = df['avg_consumption_cost'].sum()
        total_optimized_cost = df['optimized_cost'].sum()
        total_savings = df['savings'].sum()
        savings_percentage = (total_savings / total_avg_cost) * 100
        
        # Find best and worst days
        best_day = df.loc[df['savings'].idxmax()]
        worst_day = df.loc[df['savings'].idxmin()]
        
        metrics = {
            'total_avg_cost': total_avg_cost,
            'total_optimized_cost': total_optimized_cost,
            'total_savings': total_savings,
            'savings_percentage': savings_percentage,
            'best_day_savings': best_day['savings'],
            'worst_day_savings': worst_day['savings'],
            'average_daily_savings': df['savings'].mean(),
            'total_consumption': df['total_consumption'].iloc[0],
            'average_consumption': df['avg_consumption'].iloc[0]
        }
        
        self.logger.info("Savings analysis completed")
        return metrics
    
    def save_results(self, results_df: pd.DataFrame, metrics: Dict):
        """Save results to CSV and create visualization"""
        try:
            # Save results to CSV
            results_path = Path('workspaces/sandeep/warp/results')
            results_path.mkdir(parents=True, exist_ok=True)
            results_file = results_path / 'cost_savings_analysis.csv'
            results_df.to_csv(results_file)
            self.logger.info(f"Results saved to {results_file}")
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot costs
            ax1.plot(results_df.index, results_df['avg_consumption_cost'], 
                    label='Cost with Average Consumption', marker='o')
            ax1.plot(results_df.index, results_df['optimized_cost'], 
                    label='Cost with Optimized Consumption', marker='o')
            ax1.plot(results_df.index, results_df['cumulative_savings'], 
                    label='Cumulative Savings', linestyle='--')
            
            ax1.set_title('Cost Comparison and Savings Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cost ($)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot consumption
            ax2.plot(results_df.index, results_df['optimized_consumption'], 
                    label='Optimized Consumption', marker='o')
            ax2.axhline(y=self.avg_daily_consumption, color='r', linestyle='--', 
                       label='Average Consumption')
            
            ax2.set_title('Daily Consumption')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Consumption (units)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = results_path / 'cost_savings_plot.png'
            plt.savefig(plot_file)
            plt.close()
            self.logger.info(f"Plot saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def plot_savings(self, results_df: pd.DataFrame, save_path: str = None):
        """Plot the cost savings analysis"""
        self.logger.info("Creating savings visualization...")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        
        # Plot 1: Daily costs and savings
        ax1.plot(results_df['timestamp'], results_df['avg_consumption_cost'], 
                 label='Average Consumption Cost', color='blue', alpha=0.7)
        ax1.plot(results_df['timestamp'], results_df['optimized_cost'], 
                 label='Optimized Cost', color='green', alpha=0.7)
        ax1.fill_between(results_df['timestamp'], 
                         results_df['avg_consumption_cost'], 
                         results_df['optimized_cost'],
                         where=(results_df['avg_consumption_cost'] > results_df['optimized_cost']),
                         color='green', alpha=0.2, label='Savings')
        ax1.fill_between(results_df['timestamp'], 
                         results_df['avg_consumption_cost'], 
                         results_df['optimized_cost'],
                         where=(results_df['avg_consumption_cost'] <= results_df['optimized_cost']),
                         color='red', alpha=0.2, label='Losses')
        
        ax1.set_title('Daily Cost Comparison: Average vs Optimized Consumption')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cost')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative savings
        ax2.plot(results_df['timestamp'], results_df['cumulative_savings'], 
                 label='Cumulative Savings', color='purple')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        ax2.set_title('Cumulative Savings Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Savings')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def run_analysis(self, input_file: Optional[str] = None) -> Dict:
        """
        Run the complete cost savings analysis
        
        Args:
            input_file: Optional path to rolling predictions CSV file
        
        Returns:
            Dictionary with analysis metrics
        """
        try:
            self.logger.info("Starting cost savings analysis...")
            
            # Load data from file or generate dummy data
            if input_file:
                self.logger.info(f"Loading data from file: {input_file}")
                df = self.load_rolling_predictions(input_file)
            else:
                self.logger.info("No input file provided, generating dummy data")
                df = self.generate_dummy_data()
            
            # Calculate costs and savings
            results_df = self.calculate_cost_savings(df)
            
            # Analyze savings
            metrics = self.analyze_savings(results_df)
            
            # Print results
            print("\n=== Cost Savings Analysis ===")
            print(f"\nAverage Daily Consumption: {self.avg_daily_consumption} units")
            print("\nDaily Costs and Savings:")
            print(results_df[['timestamp', 'actual_price', 'predicted_price', 
                            'optimized_consumption', 'avg_consumption_cost', 
                            'optimized_cost', 'savings']].to_string())
            
            print("\nSummary Metrics:")
            print(f"Total Cost with Average Consumption: ${metrics['total_avg_cost']:.2f}")
            print(f"Total Cost with Optimized Consumption: ${metrics['total_optimized_cost']:.2f}")
            print(f"Total Savings: ${metrics['total_savings']:.2f}")
            print(f"Savings Percentage: {metrics['savings_percentage']:.1f}%")
            print(f"Average Daily Savings: ${metrics['average_daily_savings']:.2f}")
            print(f"Best Day Savings: ${metrics['best_day_savings']:.2f}")
            print(f"Worst Day Savings: {metrics['worst_day_savings']:.2f}")
            print(f"Total Consumption: {metrics['total_consumption']:.1f} units")
            print(f"Average Consumption: {metrics['average_consumption']:.1f} units")
            
            # Save results
            self.save_results(results_df, metrics)
            
            # Plot savings
            self.plot_savings(results_df)
            
            self.logger.info("Cost savings analysis completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            raise

# Example usage
if __name__ == '__main__':
    # Create analyzer instance with 1000 units daily consumption
    analyzer = CostSavingsAnalyzer(avg_daily_consumption=1000.0)
    
    # Run analysis with rolling predictions file
    input_file = "/Users/sgawde/work/eaisi-code/main-branch-01-jun/ENEXIS/src/models/model_run_results/rolling_predictions.csv"
    analyzer.run_analysis(input_file)