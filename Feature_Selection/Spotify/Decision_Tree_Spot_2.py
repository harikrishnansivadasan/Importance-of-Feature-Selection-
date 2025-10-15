from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

df = pd.read_csv('dataset.csv')

# Split the features and target variable
X = df.drop(['No','track_genre','track_id'], axis=1)
y = df['track_genre']

# Label encode categorical variables
cat_cols = ['artists','album_name','track_name','explicit'] # list of categorical column names
label_encoders = {}
for col in cat_cols:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

dt = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
dt.fit(X, y)

# Select the most important features using decision tree
selector = SelectFromModel(estimator=dt, prefit=True)
selected_features = X.columns[selector.get_support()]

# Print and save the selected features to a new file
print('Selected Features:', selected_features)
selected_cols = X.columns[selector.get_support()]
X_new = pd.DataFrame(selector.transform(X), columns=selected_cols)
X_new["track_genre"] = y
X_new.to_csv('Spotify_DecisionTree.csv', index=False)
