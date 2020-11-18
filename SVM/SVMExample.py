import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Loads pre-installed data set from sklearn
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

# Grabs data and target pre-assigned by sklearn
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.2)

classes = ['malignant' 'benign']

# Creates SVC Model/ Fits Data
# SVC = Support Vector Classification
# Look at SVC Documentation for more kernals
# kernel = function type
# C = margin (default = 1)
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

# Predicts results using model and test data
y_pred = clf.predict(x_test)

# Gets accuracy of model
# Must add parameters to get higher accuracy
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)