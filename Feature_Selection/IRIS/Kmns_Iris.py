'''from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pandas as pd

X_new = pd.read_csv("selected_features.csv")
X = X_new[["sepal length (cm)",  "sepal width (cm)" ]]
y = X_new[["species"]]

# Train a K-means classifier
clf = KMeans(n_clusters=3)
clf.fit(X)

# Predict the classes of the samples
y_pred = clf.predict(X)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)'''
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

X_new = pd.read_csv("selected_features.csv")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new[["petal length (cm)",  "sepal length (cm)"]] , X_new[["Target Variable Label"]], test_size=0.3, random_state=42)

# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict the classes of the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)

