from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
# load the iris dataset
#iris= datasets.load_iris()
X_new = pd.read_csv("Diab_RandFor.csv")
#print(X_new[["species"]])
# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new[["Glucose",  "BMI", "DiabetesPedigreeFunction", "Age" ]] , X_new[["Outcome"]], test_size=0.3)

# create a support vector machine classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# train the classifier on the training data
clf.fit(X_train, y_train)

# make predictions on the testing data
y_pred = clf.predict(X_test)

# print the accuracy score
print("Accuracy:", clf.score(X_test, y_test)*100)