"""
Football Player Rating Prediction System
=========================================
This system predicts player ratings based on match statistics using multiple machine learning algorithms.

Dataset Features:
- Minutes played
- Goals
- Assists
- Passes
- Key Passes
- Big Chances Created
- Shots
- Shots on Target
- Yellow card
- Red card

Target Variable: Player Rating (0-10 scale)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class FootballRatingPredictor:
    """
    A class to predict football player ratings using multiple ML algorithms
    """
    
    def __init__(self, csv_path='Football.csv'):
        """
        Initialize the predictor with dataset path
        
        Parameters:
        -----------
        csv_path : str
            Path to the Football.csv dataset
        """
        self.csv_path = csv_path
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.feature_names = None
        
    def load_data(self):
        """Load and display dataset information"""
        print("=" * 60)
        print("LOADING DATASET")
        print("=" * 60)
        
        self.dataset = pd.read_csv(self.csv_path)
        print(f"\nDataset Shape: {self.dataset.shape}")
        print(f"Number of samples: {self.dataset.shape[0]}")
        print(f"Number of features: {self.dataset.shape[1] - 1}")
        
        print("\nFirst 5 rows:")
        print(self.dataset.head())
        
        print("\nDataset Info:")
        print(self.dataset.info())
        
        print("\nStatistical Summary:")
        print(self.dataset.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.dataset.isnull().sum())
        
        return self.dataset
    
    def prepare_data(self, test_size=0.30, random_state=42):
        """
        Prepare data for training
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset for testing (default: 0.30)
        random_state : int
            Random seed for reproducibility (default: 42)
        """
        print("\n" + "=" * 60)
        print("PREPARING DATA")
        print("=" * 60)
        
        # Separate features and target
        X = self.dataset.iloc[:, :-1].values
        y = self.dataset.iloc[:, -1].values
        
        # Store feature names
        self.feature_names = self.dataset.columns[:-1].tolist()
        
        print(f"\nFeatures: {self.feature_names}")
        print(f"Target: {self.dataset.columns[-1]}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        
    def train_models(self):
        """Train multiple regression models"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)
        
        # Define models
        self.models = {
            'Multiple Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Machine': SVR(kernel='rbf')
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"✓ {name} trained successfully")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        results = []
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(self.X_test)
            self.predictions[name] = y_pred
            
            # Calculate metrics
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            
            # Store metrics
            self.metrics[name] = {
                'R² Score': r2,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae
            }
            
            results.append({
                'Model': name,
                'R² Score': r2,
                'RMSE': rmse,
                'MAE': mae
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('R² Score', ascending=False)
        
        print("\n" + "-" * 60)
        print("PERFORMANCE COMPARISON")
        print("-" * 60)
        print(results_df.to_string(index=False))
        
        # Find best model
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   R² Score: {results_df.iloc[0]['R² Score']:.4f}")
        
        return results_df
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Model Performance Comparison
        ax1 = plt.subplot(2, 3, 1)
        models_names = list(self.metrics.keys())
        r2_scores = [self.metrics[m]['R² Score'] for m in models_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models_names)))
        bars = ax1.barh(models_names, r2_scores, color=colors)
        ax1.set_xlabel('R² Score', fontsize=10)
        ax1.set_title('Model Performance (R² Score)', fontsize=12, fontweight='bold')
        ax1.set_xlim([0, 1])
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        # 2. RMSE Comparison
        ax2 = plt.subplot(2, 3, 2)
        rmse_scores = [self.metrics[m]['RMSE'] for m in models_names]
        ax2.barh(models_names, rmse_scores, color=colors)
        ax2.set_xlabel('RMSE (Lower is Better)', fontsize=10)
        ax2.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
        for i, (model, rmse) in enumerate(zip(models_names, rmse_scores)):
            ax2.text(rmse, i, f'{rmse:.4f}', ha='left', va='center', fontsize=9)
        
        # 3. MAE Comparison
        ax3 = plt.subplot(2, 3, 3)
        mae_scores = [self.metrics[m]['MAE'] for m in models_names]
        ax3.barh(models_names, mae_scores, color=colors)
        ax3.set_xlabel('MAE (Lower is Better)', fontsize=10)
        ax3.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        for i, (model, mae) in enumerate(zip(models_names, mae_scores)):
            ax3.text(mae, i, f'{mae:.4f}', ha='left', va='center', fontsize=9)
        
        # 4. Actual vs Predicted - Best Model
        best_model = max(self.metrics.items(), key=lambda x: x[1]['R² Score'])[0]
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(self.y_test, self.predictions[best_model], alpha=0.6, color='blue')
        ax4.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax4.set_xlabel('Actual Rating', fontsize=10)
        ax4.set_ylabel('Predicted Rating', fontsize=10)
        ax4.set_title(f'Actual vs Predicted - {best_model}', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Prediction Distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(self.y_test, bins=20, alpha=0.5, label='Actual', color='green', edgecolor='black')
        ax5.hist(self.predictions[best_model], bins=20, alpha=0.5, label='Predicted', 
                color='orange', edgecolor='black')
        ax5.set_xlabel('Player Rating', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.set_title('Distribution of Ratings', fontsize=12, fontweight='bold')
        ax5.legend()
        
        # 6. Feature Importance (for Random Forest)
        if 'Random Forest' in self.models:
            ax6 = plt.subplot(2, 3, 6)
            rf_model = self.models['Random Forest']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            ax6.barh(range(len(importances)), importances[indices], color='teal')
            ax6.set_yticks(range(len(importances)))
            ax6.set_yticklabels([self.feature_names[i] for i in indices], fontsize=9)
            ax6.set_xlabel('Importance', fontsize=10)
            ax6.set_title('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualizations saved as 'model_comparison.png'")
        plt.show()
    
    def predict_rating(self, stats):
        """
        Predict player rating using the best model
        
        Parameters:
        -----------
        stats : list
            Player statistics [Minutes, Goals, Assists, Passes, Key Passes, 
                              Big Chances, Shots, Shots on Target, Yellow, Red]
        
        Returns:
        --------
        dict : Predictions from all models
        """
        print("\n" + "=" * 60)
        print("MAKING PREDICTION")
        print("=" * 60)
        
        stats_array = np.array(stats).reshape(1, -1)
        
        print("\nInput Statistics:")
        for feature, value in zip(self.feature_names, stats):
            print(f"  {feature}: {value}")
        
        print("\nPredicted Ratings:")
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(stats_array)[0]
            predictions[name] = pred
            print(f"  {name}: {pred:.2f}")
        
        return predictions
    
    def save_results(self, filename='predictions_comparison.csv'):
        """Save predictions and actual values to CSV"""
        results_df = pd.DataFrame({
            'Actual': self.y_test
        })
        
        for name, pred in self.predictions.items():
            results_df[f'Predicted_{name}'] = pred
        
        results_df.to_csv(filename, index=False)
        print(f"\n✓ Results saved to '{filename}'")


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOOTBALL PLAYER RATING PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize predictor
    predictor = FootballRatingPredictor('Football.csv')
    
    # Load and explore data
    predictor.load_data()
    
    # Prepare data
    predictor.prepare_data(test_size=0.30, random_state=42)
    
    # Train models
    predictor.train_models()
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Visualize results
    predictor.visualize_results()
    
    # Save results
    predictor.save_results()
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    example_stats = [90, 0, 0, 32, 2, 1, 0, 0, 1, 0]
    predictor.predict_rating(example_stats)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)