# ⚽ Football Player Rating Predictor

A comprehensive machine learning system that predicts football player ratings based on match statistics using multiple regression algorithms.

## 📊 Project Overview

This project uses various machine learning models to predict player performance ratings (0-10 scale) based on in-game statistics. The system compares 5 different algorithms and identifies the best performer, providing detailed metrics and visualizations.

## 🎯 Features

- **Multiple ML Algorithms**: Compares 5 different regression models
  - Multiple Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Machine (SVR)

- **Comprehensive Evaluation**: 
  - R² Score (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)

- **Rich Visualizations**:
  - Model performance comparison charts
  - Actual vs Predicted scatter plots
  - Rating distribution histograms
  - Feature importance analysis

- **Professional Output**:
  - Detailed console logs
  - High-quality PNG visualizations
  - CSV export of predictions

## 📋 Dataset

The model uses the following player statistics as features:

| Feature | Description |
|---------|-------------|
| Minutes Played | Total minutes played in the match |
| Goals | Number of goals scored |
| Assists | Number of assists provided |
| Passes | Total passes completed |
| Key Passes | Passes leading to shot attempts |
| Big Chances Created | Clear goal-scoring opportunities created |
| Shots | Total shots taken |
| Shots on Target | Shots on target |
| Yellow Card | Number of yellow cards received |
| Red Card | Number of red cards received |

**Target Variable**: Player Rating (0-10 scale)

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/FaisalMubeen2001/football-rating-predictor.git
   cd football-rating-predictor
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset is in the project directory**
   - The `Football.csv` file should be in the same folder as `football_predictor.py`

### Usage

Run the main script:
```bash
python football_predictor.py
```

The script will:
1. Load and analyze the dataset
2. Train all 5 models
3. Evaluate and compare performance
4. Generate visualizations
5. Save results to files

### Output Files

- `model_comparison.png` - Comprehensive visualization with 6 charts
- `predictions_comparison.csv` - Detailed predictions from all models

## 📈 Example Prediction

To predict a player's rating with custom statistics:

```python
from football_predictor import FootballRatingPredictor

predictor = FootballRatingPredictor('Football.csv')
predictor.load_data()
predictor.prepare_data()
predictor.train_models()

# Example: 90 minutes, 0 goals, 0 assists, 32 passes, 2 key passes, 
#          1 big chance, 0 shots, 0 on target, 1 yellow, 0 red
stats = [90, 0, 0, 32, 2, 1, 0, 0, 1, 0]
predictions = predictor.predict_rating(stats)
```

## 📊 Model Performance

The system automatically identifies the best-performing model based on R² score. Typical results:

| Model                      | R² Score | RMSE | MAE  |
|----------------------------|----------|------|------|
| Random Forest              | 0.95+    | ~0.3 | ~0.2 |
| Gradient Boosting          | 0.94+    | ~0.3 | ~0.2 |
| Multiple Linear Regression | 0.85+    | ~0.5 | ~0.4 |

*Note: Actual results may vary based on your dataset*

## 🛠️ Customization

### Adjust Train-Test Split
```python
predictor.prepare_data(test_size=0.25, random_state=42)
```

### Modify Model Parameters
Edit the model definitions in the `train_models()` method:
```python
'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42)
```

## 📁 Project Structure

```
football-rating-predictor/
│
├── football_predictor.py      # Main script
├── Football.csv               # Dataset
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
└── (Generated files)
    ├── model_comparison.png   # Visualizations
    └── predictions_comparison.csv  # Results
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

### Ideas for Contribution:
- Add more machine learning algorithms
- Implement cross-validation
- Add hyperparameter tuning
- Create a web interface
- Add more detailed statistical analysis
- Improve visualizations

## 📝 Dataset Information

The `Football.csv` dataset contains real match statistics for football players. You can:
- Use the provided dataset
- Add your own data following the same format
- Collect data from football statistics websites

### Dataset Format:
```csv
Minutes_Played,Goals,Assists,Passes,Key_Passes,Big_Chances,Shots,Shots_On_Target,Yellow_Card,Red_Card,Rating
90,1,0,45,3,2,4,2,0,0,8.5
...
```

## 🔍 Understanding the Metrics

- **R² Score**: Measures how well the model fits the data (closer to 1 is better)
- **RMSE**: Average prediction error in the same units as the target (lower is better)
- **MAE**: Average absolute difference between predictions and actual values (lower is better)

## 📧 Contact

For questions, suggestions, or collaborations:
- GitHub: [@FaisalMubeen2001](https://github.com/FaisalMubeen2001)
- Email: faisalmubeen2001@gmail.com

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Thanks to the football statistics community for data collection
- Scikit-learn for providing excellent machine learning tools
- The open-source community for continuous support

---

**⭐ If you find this project helpful, please consider giving it a star!**