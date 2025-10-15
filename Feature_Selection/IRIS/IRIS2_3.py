from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Load iris dataset
iris = load_iris()

# Create PCA object with 2 components
pca = PCA(n_components=0.85)

# Fit PCA to the data
iris_pca = pca.fit_transform(iris.data)

# Get the indices of the selected features
selected_feature_indices = pca.components_.argsort()[:, ::-1][:, :2].ravel()

# Get the names of the selected features
selected_feature_names = np.array(iris.feature_names)[selected_feature_indices]

# Access the selected features in the original data
selected_features = iris.data[:, selected_feature_indices]

# Create a DataFrame with the selected features and target variable
df = pd.DataFrame(data=selected_features, columns=selected_feature_names)
df['Target Variable'] = iris.target
df['Target Variable Label'] = df['Target Variable'].apply(lambda x: iris.target_names[x])

# Write the DataFrame to a new file
df.to_csv('selected_features.csv', index=False)

# Print the selected features
print(selected_features)
