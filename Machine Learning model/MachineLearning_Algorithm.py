import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

attention_data = pd.read_csv('Dataset.csv', delimiter=",")#, names=['WKc', 'WKr', 'WRc', 'WRr', 'BKc', 'BKr', 'Optimal depth'])
attention_data = pd.DataFrame(attention_data)
target=attention_data['Level_attention']
features=attention_data.drop('Level_attention', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, train_size=0.7, random_state=0)

##  Decision Tree Classifier
accuracy = []
for i in range(1, 40):
     # Create Decision Tree classifer object
     clf = DecisionTreeClassifier(criterion="entropy", max_depth=i)
     # Train Decision Tree Classifer
     clf = clf.fit(X_train, y_train)
     # Predict the response for test dataset
     y_pred = clf.predict(X_test)
     # Model Accuracy, how often is the classifier correct?
     accuracy.append(metrics.accuracy_score(y_test, y_pred))
     # print(clf.tree_.max_depth)
plt.plot(range(1,40),accuracy, color='k',  linewidth = 3, marker='o',markerfacecolor='blue',markersize=6)
plt.ylabel('Accuracy')
plt.xlabel('Depth')
plt.title('Accuracy vs Depth')
plt.show()

clf = DecisionTreeClassifier(criterion="entropy", max_depth=12)
## Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
##Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print('Decision Tree Accuracy: ',metrics.accuracy_score(y_test, y_pred))

##  KNN Classifier
accuracy=[]
for i in range(1,13):
   knn = KNeighborsClassifier(n_neighbors = i)
   knn = knn.fit(X_train, y_train)
   y_pred = knn.predict(X_test)
   accuracy.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(range(1, 13), accuracy, color='k', linewidth=3, marker='o', markerfacecolor='blue', markersize=6)
plt.ylabel('Accuracy')
plt.xlabel('Neighbors')
plt.title('Accuracy vs Neighbors')
plt.show()
knn = KNeighborsClassifier(n_neighbors = 3)
knn = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('KNN Accuracy: ',metrics.accuracy_score(y_test, y_pred))