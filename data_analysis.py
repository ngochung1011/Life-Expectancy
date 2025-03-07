import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    """Generate a heatmap to visualize feature correlations."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_histogram(df, column):
    """Plot histogram of a given feature."""
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("processed_life_expectancy.csv")
    plot_correlation_heatmap(df)
    plot_histogram(df, "Life expectancy")
