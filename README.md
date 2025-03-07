# Life Expectancy Prediction using Linear Regression

## 📌 Introduction
Predicting life expectancy is an important task in public health and policy planning. This project utilizes **Linear Regression** to estimate life expectancy based on factors such as **GDP, Schooling, and Adult Mortality**. The dataset is sourced from the **World Health Organization (WHO)** and the **United Nations**.

---

## 📑 Project Overview
### Objectives:
- **Preprocess** life expectancy data by handling missing values and normalizing features.
- **Analyze correlations** between various socio-economic and health indicators.
- **Train a linear regression model** to predict life expectancy.
- **Evaluate model performance** using MAE and MSE.

### Key Features:
- Uses **GDP, Schooling, and Adult Mortality** as key predictors.
- Implements **Linear Regression with Scikit-Learn**.
- Visualizes data distributions and correlations.

---

## 📂 Project Structure
```
├── data/                      # Raw and processed data
│   ├── life_expectancy.csv    # Original dataset
│   ├── processed_life_expectancy.csv # Preprocessed data
│
├── models/                    # Trained models
│   ├── life_expectancy_model.pkl # Saved regression model
│
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data cleaning & feature engineering
│   ├── data_analysis.py       # Data visualization & correlation analysis
│   ├── linear_regression.py   # Training the regression model
│   ├── model_evaluation.py    # Evaluating model performance
│
├── results/                   # Stores evaluation metrics
│   ├── predictions.npy        # Predicted values
│   ├── actual_values.npy      # Actual life expectancy
│   ├── performance_metrics.txt # Model performance summary
│
├── README.md                  # Project documentation
```

---

## ⚙️ Installation
To run this project, install the required dependencies:
```sh
pip install numpy pandas seaborn matplotlib scikit-learn
```
For a virtual environment:
```sh
python -m venv env
source env/bin/activate  # On Mac/Linux
# OR
env\Scripts\activate  # On Windows
pip install -r requirements.txt
```

---

## 🚀 How to Use

### 1️⃣ Data Preprocessing
```python
from src.data_preprocessing import load_and_clean_data, scale_features
data = load_and_clean_data("data/life_expectancy.csv")
data = scale_features(data, ['GDP', 'Schooling', 'Adult Mortality'])
data.to_csv("data/processed_life_expectancy.csv", index=False)
```

### 2️⃣ Data Analysis
```python
from src.data_analysis import plot_correlation_heatmap, plot_histogram
import pandas as pd

df = pd.read_csv("data/processed_life_expectancy.csv")
plot_correlation_heatmap(df)
plot_histogram(df, "Life expectancy")
```

### 3️⃣ Train the Model
```python
from src.linear_regression import train_model

df = pd.read_csv("data/processed_life_expectancy.csv")
trained_model = train_model(df)
```

### 4️⃣ Evaluate Model Performance
```python
from src.model_evaluation import evaluate_model
mae, mse = evaluate_model()
print(f"MAE: {mae}, MSE: {mse}")
```

---

## 📊 Model Performance
| Model                 | MAE   | MSE   |
|-----------------------|------ |------ |
| Linear Regression     | 2.15  | 5.42  |

The model achieves an accuracy of **86.66%**, showing a strong correlation between **education, economic factors, and life expectancy**.

---

## 🔥 Future Improvements
- **Include additional health indicators** such as healthcare access, environmental factors.
- **Try advanced regression techniques** like Ridge, Lasso, or XGBoost.
- **Optimize feature selection** using recursive elimination.

---

## 🤝 Contributors
📌 **Project Lead:** Đặng Ngọc Hưng  
📌 **Supervisors:** ThS.Hà Văn Thảo

If you find this project useful, feel free to fork and contribute! 🚀

