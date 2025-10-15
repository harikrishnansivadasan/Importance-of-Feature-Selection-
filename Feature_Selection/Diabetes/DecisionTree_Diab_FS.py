from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
#from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

df = pd.read_csv('diabetes.csv')

# Split the features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']


# Create a Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42)

# Use SelectFromModel to select the most important features
sfm = SelectFromModel(dtc, threshold='median')
X_selected = sfm.fit_transform(X, y)

# Print the selected features
selected_features = np.array(X.columns)[sfm.get_support()]
print("Selected Features: ", selected_features)
X_new = pd.DataFrame(X_selected, columns=selected_features)
X_new["Outcome"] = y
X_new.to_csv('Diab_DT.csv', index=False)

print(X_new)