from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

df = pd.read_csv('dataset.csv')

# Split the features and target variable
X = df.drop(['No','track_genre','track_id'], axis=1)
y = df['track_genre']

# One-hot encode categorical variables
cat_cols = ['artists','album_name','track_name','explicit'] # list of categorical column names
ohe = OneHotEncoder()
X_cat = ohe.fit_transform(X[cat_cols])
X_cat_df = pd.DataFrame(X_cat.toarray(), columns=ohe.get_feature_names(cat_cols))

# Combine one-hot encoded categorical variables with numerical variables
num_cols = [col for col in X.columns if col not in cat_cols]
X_num = X[num_cols]
X = pd.concat([X_num, X_cat_df], axis=1)

dt = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
dt.fit(X, y)

# Select the most important features using decision tree
selector = SelectFromModel(estimator=dt, prefit=True)
selected_features = X.columns[selector.get_support()]

# Print and save the selected features to a new file
print('Selected Features:', selected_features)
X_new = selector.transform(X[selected_features])
X_new = pd.DataFrame(X_new, columns=selected_features)
X_new["track_genre"] = y
X_new.to_csv('Spotify_Dum.csv', index=False)
