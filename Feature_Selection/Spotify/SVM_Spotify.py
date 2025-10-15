from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
# load the iris dataset
#iris= datasets.load_iris()

'''df = pd.read_csv('Spotify_DecisionTree.csv')

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("Spotify_DecisionTree.csv")

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
selected_df.to_csv("output_file.csv", index=False)'''

X_new = pd.read_csv("output_file.csv",nrows=200)
'''gen = 
print(len(gen), "\n", gen)'''
# split the dataset into training and testing sets
x = X_new.drop(X_new.columns[len(X_new.columns)-1],axis=1)
y = X_new[X_new.columns[len(X_new.columns)-1]]
X_train, X_test, y_train, y_test = train_test_split(x , y, test_size=0.3)

# create a support vector machine classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# train the classifier on the training data
clf.fit(X_train, y_train)

# make predictions on the testing data
y_pred = clf.predict(X_test)

# print the accuracy score
print("Accuracy:", clf.score(X_test, y_test)*100)