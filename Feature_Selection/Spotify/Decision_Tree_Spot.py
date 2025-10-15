from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
#from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

'''df = pd.read_csv('dataset.csv')

# Split the features and target variable
X = df.drop('track_genre', axis=1)
y = df['track_genre']


# Create a Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X, y)
# Use SelectFromModel to select the most important features
#sfm = SelectFromModel(dtc, threshold='median')
X_selected = SelectFromModel(estimator=dtc, prefit=True)

# Print the selected features
selected_features = np.array(X.columns)[sfm.get_support()]
print("Selected Features: ", selected_features)
X_new = pd.DataFrame(X_selected, columns=selected_features)
X_new["track_genre"] = y
X_new.to_csv('Spotify_RandFor.csv', index=False)

print(X_new)

dt = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
dt.fit(X, y)

# Select the most important features using decision tree
selector = SelectFromModel(estimator=dt, prefit=True)
selected_features = X.columns[selector.get_support()]

# Print and save the selected features to a new file
print('Selected Features:', selected_features)
X_new = pd.DataFrame(selector, columns=selected_features)
X_new["track_genre"] = y
X_new.to_csv('Spotify_RandFor.csv', index=False)
#selected_features.to_csv('selected_features.csv', index=False)'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd

df = pd.read_csv('dataset.csv')

# Split the features and target variable
X = df.drop('track_genre', axis=1)
y = df['track_genre']

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
X_new.to_csv('Spotify_RandFor.csv', index=False)
#selected_features.to_csv('selected_features.csv', index=False)
