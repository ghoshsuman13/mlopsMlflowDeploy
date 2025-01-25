# generate_dataset.py
import pandas as pd
from sklearn.datasets import make_regression

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=0.2, random_state=42)
data = pd.DataFrame({'feature': X.flatten(), 'target': y})

# Save dataset
data.to_csv("data/dataset.csv", index=False)
print("Dataset saved to data/dataset.csv")
