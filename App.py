
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ClassificationReport

weather_data = pd.read_csv('weatherAUS.csv', header=0)
# print(stars_data.dtypes)

cols = []
for col in weather_data.columns:
    if col != 'RainTodayNum' and col != 'RainToday' and col != '1' and col != '2' and col != '3' and col != '4' and \
            col != '5' and col != '6' and col != '7' and col != '8' and col != '9' and col != '10'and col != '11'\
            and col != '12' and col != '13' and col != '14' and col != '15' and col != '16' and col != '17':
        cols.append(col)
# dropping the 'Outcome'column
data = weather_data[cols]

# assigning the Outcome column as target
target = weather_data['RainTodayNum']
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.50, random_state=4)


#                                  ~~~~~~~~~~~*Naive-Bayes accuracy*~~~~~~~~~~
# create an object of the type GaussianNB
gnb = GaussianNB()
# train the algorithm on training data and predict using the testing data
gnb.fit(X_train, Y_train)
pred_gnb = gnb.predict(X_test)
# print the accuracy score of the model
print("Naive-Bayes accuracy : ", accuracy_score(Y_test, pred_gnb, normalize=True))
print(confusion_matrix(Y_test, pred_gnb))
print(classification_report(Y_test, pred_gnb))




#                                  ~~~~~~~~~~~*K-Neighbors Classifier*~~~~~~~~~~

'''
# create object of the Classifier
knn = KNeighborsClassifier(n_neighbors=5)
# Train the algorithm
knn.fit(X_train, Y_train)
# predict the response
pred_knn = knn.predict(X_test)
# evaluate accuracy
print("KNeighbors accuracy score : ", accuracy_score(Y_test, pred_knn))
print(confusion_matrix(Y_test, pred_knn))
print(classification_report(Y_test, pred_knn))
'''

'''k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(Y_test, y_pred))
# allow plots to appear within the notebook

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')'''

#                                  ~~~~~~~~~~~*Logistic regression Classifier*~~~~~~~~~~

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
pred_logreg = logreg.predict(X_test)
# compare actual response values (y_test) with predicted response values (y_pred)
print("LogisticRegression accuracy score : ",accuracy_score(Y_test, pred_logreg))
print(confusion_matrix(Y_test, pred_logreg))
print(classification_report(Y_test, pred_logreg))

#                                  ~~~~~~~~~~~*Decision Trees Classifier*~~~~~~~~~~

Tree=tree.DecisionTreeClassifier()
Tree.fit(X_train, Y_train)
pred_Tree = Tree.predict(X_test)
print("Decision Trees accuracy score : ",accuracy_score(Y_test, pred_Tree))
print(confusion_matrix(Y_test, pred_Tree))
print(classification_report(Y_test, pred_Tree))


#                                  ~~~~~~~~~~~*Naive-Bayes plot*~~~~~~~~~~
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(gnb, classes=['Not Rain today', ' Rain today'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g_gnb = visualizer.poof()  # Draw/show/poof the data

#                                  ~~~~~~~~~~~*K-Neighbors Classifier plot*~~~~~~~~~~
'''# Instantiate the classification model and visualizer
visualizer = ClassificationReport(knn, classes=['Not Rain today', ' Rain today'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g3 = visualizer.poof()  # Draw/show/poof the data'''

#                                  ~~~~~~~~~~~*Logistic regression*~~~~~~~~~~
visualizer = ClassificationReport(logreg, classes=['Not Rain today', ' Rain today'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g_logreg = visualizer.show()

#                                  ~~~~~~~~~~~*Decision Trees*~~~~~~~~~~
visualizer = ClassificationReport(Tree, classes=['Not Rain today', ' Rain today'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g_Tree= visualizer.poof()

