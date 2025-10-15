from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

# Split the features and target variable
X = df.drop(['No','track_genre','track_id','album_name','track_name'], axis=1)
y = df['track_genre']


# Label encode categorical variables
cat_cols = ['artists','explicit'] # list of categorical column names
label_encoders = {}
for col in cat_cols:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

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
X_new["track_genre"] = y
X_new.to_csv('Spotify_RandFor.csv', index=False)
#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#

#Knn_Classifier
X_new = pd.read_csv("Spotify_RandFor.csv")
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
'''# Calculate the confusion matrix
cm_k = confusion_matrix(y_test_k, y_pred_k)

# Plot the confusion matrix
plt.imshow(cm_k, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for KNN Classifier')
plt.colorbar()
plt.xticks([0,1], labels=['Negative','Positive'])
plt.yticks([0,1], labels=['Negative','Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()'''
#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#

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
'''# Calculate the confusion matrix
cm_dt = confusion_matrix(y_test_dt, y_pred_dt)

# Plot the confusion matrix
plt.imshow(cm_dt, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.colorbar()
plt.xticks([0,1], labels=['Negative','Positive'])
plt.yticks([0,1], labels=['Negative','Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()'''
#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#
#SVM
# Read the CSV file into a pandas DataFrame
df = pd.read_csv("Spotify_RandFor.csv")

# Group the DataFrame by the class label column
grouped = df.groupby("track_genre")

# Create an empty DataFrame to store the selected rows
selected_df = pd.DataFrame()

# Loop through each group
for name, group in grouped:
    # Select the first 10 rows of the group
    group = group.iloc[:10]
    
    # Append the selected rows to the new DataFrame
    selected_df = pd.concat([selected_df, group])

# Write the selected rows to a new CSV file
selected_df.to_csv("output_file_RF.csv", index=False)
X_new_s = pd.read_csv("output_file_RF.csv",nrows=200)
# split the dataset into training and testing sets
x = X_new_s.drop(X_new.columns[len(X_new_s.columns)-1],axis=1)
y = X_new_s[X_new_s.columns[len(X_new_s.columns)-1]]
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(x , y, test_size=0.3)

# create a support vector machine classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# train the classifier on the training data
clf.fit(X_train_svm, y_train_svm)

# make predictions on the testing data
y_pred_svm = clf.predict(X_test_svm)

# print the accuracy score
print("Accuracy of SVM:", clf.score(X_test_svm, y_test_svm)*100)
'''# Calculate the confusion matrix
cm_svm = confusion_matrix(y_test_svm, y_pred_svm)

# Plot the confusion matrix
plt.imshow(cm_svm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for SVM Classifier')
plt.colorbar()
plt.xticks([0,1], labels=['Negative','Positive'])
plt.yticks([0,1], labels=['Negative','Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()'''