from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFECV

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target
dtc = DecisionTreeClassifier(random_state=42)

# Use SelectFromModel to select the most important features
sfm = SelectFromModel(dtc, threshold='median')
X_selected = sfm.fit_transform(X, y)


# Print the selected features
selected_features = np.array(iris.feature_names)[sfm.get_support()]
print("Selected Features: ", selected_features)
X_new = pd.DataFrame(X_selected, columns=selected_features)
X_new["Species"]=iris["target_names"][y]
X_new.to_csv('iris_dt.csv', index=False)

#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

#Knn
X_new = pd.read_csv("iris_dt.csv")
x = X_new.drop(X_new.columns[len(X_new.columns)-1],axis=1)
y = X_new[X_new.columns[len(X_new.columns)-1]]
#print(x,"\n",y)
# Split the dataset into training and testing sets
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(x, y, test_size=0.3, random_state=42)

# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=2)

# Train the classifier on the training data
knn.fit(X_train_k, y_train_k)

# Predict the classes of the test set
y_pred_k = knn.predict(X_test_k)

# Calculate the accuracy of the classifier
accuracy_k = accuracy_score(y_test_k, y_pred_k)
print("Accuracy of KNN:", accuracy_k*100)
# Calculate the confusion matrix
cm_k = confusion_matrix(y_test_k, y_pred_k, labels=['setosa', 'versicolor', 'virginica'])

# Plot the confusion matrix
plt.imshow(cm_k, cmap=plt.cm.Blues,vmin=0,vmax=35)
for i in range(cm_k.shape[0]):
    for j in range(cm_k.shape[1]):
        plt.text(j, i, str(cm_k[i,j]), horizontalalignment='center', verticalalignment='center')

plt.title('IRIS: (DT & KNN)')
plt.colorbar()
plt.xticks([0,1,2], labels=['setosa', 'versicolor', 'virginica'])
plt.yticks([0,1,2], labels=['setosa', 'versicolor', 'virginica'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('IRIS_DT_KNN.png')
plt.show()


#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

#Decision Tree
# Split the dataset into training and testing sets
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(x, y, test_size=0.7)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train_dt, y_train_dt)

# Predict the classes of the test set
y_pred_dt = clf.predict(X_test_dt)

# Evaluate the accuracy of the classifier
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
print("Accuracy of Decision Tree Classifier:", accuracy_dt*100)
# Calculate the confusion matrix
cm_dt = confusion_matrix(y_test_dt, y_pred_dt, labels=['setosa', 'versicolor', 'virginica'])

# Plot the confusion matrix
plt.imshow(cm_dt, cmap=plt.cm.Blues,vmin=0,vmax=35)
for i in range(cm_dt.shape[0]):
    for j in range(cm_dt.shape[1]):
        plt.text(j, i, str(cm_dt[i,j]), horizontalalignment='center', verticalalignment='center')

plt.title('IRIS: (DT & DT)')
plt.colorbar()
plt.xticks([0,1,2], labels=['setosa', 'versicolor', 'virginica'])
plt.yticks([0,1,2], labels=['setosa', 'versicolor', 'virginica'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('IRIS_DT_DT.png')
plt.show()
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
#SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(x,y, test_size=0.3)
# create a support vector machine classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# train the classifier on the training data
clf.fit(X_train_svm, y_train_svm)

# make predictions on the testing data
y_pred_svm = clf.predict(X_test_svm)

# print the accuracy score
print("Accuracy of SVM:", clf.score(X_test_svm, y_test_svm)*100)
# Calculate the confusion matrix
cm_svm = confusion_matrix(y_test_svm, y_pred_svm, labels=['setosa', 'versicolor', 'virginica'])

# Plot the confusion matrix
plt.imshow(cm_svm, cmap=plt.cm.Blues,vmin=0,vmax=35)
for i in range(cm_svm.shape[0]):
    for j in range(cm_svm.shape[1]):
        plt.text(j, i, str(cm_svm[i,j]), horizontalalignment='center', verticalalignment='center')

plt.title('IRIS: (DT & SVM)')
plt.colorbar()
plt.xticks([0,1,2], labels=['setosa', 'versicolor', 'virginica'])
plt.yticks([0,1,2], labels=['setosa', 'versicolor', 'virginica'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('IRIS_DT_SVM.png')
plt.show()
