from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model():
    """Evaluate model performance using MAE and MSE."""
    predictions = np.load("predictions.npy")
    actual_values = np.load("y_test.npy")
    
    mae = mean_absolute_error(actual_values, predictions)
    mse = mean_squared_error(actual_values, predictions)
    return mae, mse

if __name__ == "__main__":
    mae, mse = evaluate_model()
    print(f"MAE: {mae}, MSE: {mse}")