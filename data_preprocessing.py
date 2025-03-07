import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    """Load dataset, handle missing values, and encode categorical variables."""
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df['Status'] = df['Status'].map({'Developing': 0, 'Developed': 1})
    return df

def scale_features(df, columns):
    """Normalize selected features using StandardScaler."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

if __name__ == "__main__":
    data = load_and_clean_data("life_expectancy.csv")
    scaled_data = scale_features(data, ['GDP', 'Schooling', 'Adult Mortality'])
    scaled_data.to_csv("processed_life_expectancy.csv", index=False)
    print("Data preprocessing completed.")
