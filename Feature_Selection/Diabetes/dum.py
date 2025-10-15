from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('diabetes.csv')

# Split the features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Create PCA object with 4 components
pca = PCA(n_components=4)

# Fit PCA to the data
X_pca = pca.fit_transform(X)

# Get the indices of the selected features
selected_feature_indices = pca.components_.argsort()[:, ::-1][:, :4].ravel()

# Get the names of the selected features
selected_feature_names = np.array(X.columns)[selected_feature_indices]

# Access the selected features in the original data
selected_features = X.iloc[:, selected_feature_indices]

# Create a DataFrame with the selected features and target variable
df_selected = pd.concat([selected_features, y], axis=1)
df_selected.columns = list(selected_feature_names) + ['Outcome']
df_selected['Target_Variable_Label'] = df_selected['Target_Variable'].apply(lambda x: df['Target_Variable'][x])

# Write the DataFrame to a new file
df_selected.to_csv('selected_features_dum.csv', index=False)

# Print the selected features
print(selected_features)
