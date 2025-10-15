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


df = pd.read_csv('WineQT.csv')

# Split the features and target variable
X = df.drop('quality', axis=1)
y = df['quality']

# Create a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Use SelectFromModel to select the most important features
sfm = SelectFromModel(rfc, threshold='median')
X_selected = sfm.fit_transform(X, y)
print(type(sfm))
# Print the selected features
selected_features = np.array(X.columns)[sfm.get_support()]
print("Selected Features: ", selected_features)
X_new = pd.DataFrame(X_selected, columns=selected_features)
X_new["quality"] = y
X_new.to_csv('Wine_RandFor.csv', index=False)

#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

#Knn
X_new = pd.read_csv("Wine_RandFor.csv")
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
#Create the confusion matrix
cm_k = confusion_matrix(y_test_k, y_pred_k)

# Visualize the confusion matrix
plt.figure(figsize=(8,8))
plt.imshow(cm_k, interpolation='nearest', cmap='Blues',vmin=0,vmax=225)
for i in range(cm_k.shape[0]):
    for j in range(cm_k.shape[1]):
        plt.text(j, i, str(cm_k[i,j]), horizontalalignment='center', verticalalignment='center')
plt.title('WINE:(RF&KNN)', size=15)
plt.colorbar()
tick_marks = np.arange(len(set(y)))
plt.xticks(tick_marks, sorted(set(y)), size=10)
plt.yticks(tick_marks, sorted(set(y)), size=10)
plt.xlabel('Predicted label', size=12)
plt.ylabel('True label', size=12)
plt.tight_layout()
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

cm_dt = confusion_matrix(y_test_dt, y_pred_dt)
# Visualize the confusion matrix
plt.figure(figsize=(8,8))
plt.imshow(cm_dt, interpolation='nearest', cmap='Blues',vmin=0,vmax=225)
for i in range(cm_dt.shape[0]):
    for j in range(cm_dt.shape[1]):
        plt.text(j, i, str(cm_dt[i,j]), horizontalalignment='center', verticalalignment='center')
plt.title('WINE:(RF & DT)', size=15)
plt.colorbar()
tick_marks = np.arange(len(set(y)))
plt.xticks(tick_marks, sorted(set(y)),size=10)
plt.yticks(tick_marks, sorted(set(y)), size=10)
plt.xlabel('Predicted label', size=12)
plt.ylabel('True label', size=12)
plt.tight_layout()

plt.show()

#---------------------------------------------------------------------#
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
# Create the confusion matrix
cm_svm = confusion_matrix(y_test_svm, y_pred_svm)

# Visualize the confusion matrix
plt.figure(figsize=(8,8))
plt.imshow(cm_svm, interpolation='nearest', cmap='Blues',vmin=0,vmax=225)
for i in range(cm_svm.shape[0]):
    for j in range(cm_svm.shape[1]):
        plt.text(j, i, str(cm_svm[i,j]), horizontalalignment='center', verticalalignment='center')

plt.title('WINE:(RF & SVM)', size=15)
plt.colorbar()
tick_marks = np.arange(len(set(y)))
plt.xticks(tick_marks, sorted(set(y)),  size=10)
plt.yticks(tick_marks, sorted(set(y)), size=10)
plt.xlabel('Predicted label', size=12)
plt.ylabel('True label', size=12)
plt.tight_layout()
plt.show()

