from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def train_model(df):
    """Train a linear regression model to predict life expectancy."""
    X = df[['GDP', 'Schooling', 'Adult Mortality']]
    y = df['Life expectancy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    np.save("predictions.npy", predictions)
    np.save("y_test.npy", y_test)
    return model

if __name__ == "__main__":
    df = pd.read_csv("processed_life_expectancy.csv")
    trained_model = train_model(df)
    print("Model training completed.")
