
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

X_new = pd.read_csv("Diab_RandFor.csv")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new[["Glucose",  "BMI", "DiabetesPedigreeFunction", "Age" ]] , X_new[["Outcome"]], test_size=0.3, random_state=42)

# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict the classes of the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)

