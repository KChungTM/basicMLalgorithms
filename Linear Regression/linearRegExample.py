import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Import data set and separate data with delimiter = ";"
data = pd.read_csv("student-mat.csv", sep=";")

# These are attributes of data set
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Predict is our "label" (What we are trying to get)
predict = "G3"

# This will remove the predict value
# - X = Data you want to use
# - y = Resulting value you want
# - The "1" indicates axis being dropped (x=0,y=1)
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Takes attributes and splits into 4 arrays
# - X_train is a section of the X array
# - y_train is a section of the y array
# - test will test the accuracy of algorithm
# - Needs to train with split data because using
#   the full data set would lead to inaccurate results
# - test-size = percent we are splitting off into test samples
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Loop to find most accurate model
'''
best_acc = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # Training code
    linear = linear_model.LinearRegression()
    # Fits our data and stores in linear
    linear.fit(x_train, y_train)
    # Returns a value that indicates accuracy of our model
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best_acc:
        best_acc = acc
        # Saving model to use on future data sets
        # - Save with pickle file name
        # - wb mode lets you write a new file
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

print("Best accuracy: \n", best_acc)
'''

# Imports/Loads pickle file
# - After getting an accurate model you
#   can comment out the training loop
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Prints y=mx+b values (This is a line in 5D Space)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# Use linear model created to predict from untrained data
predictions = linear.predict(x_test)

# Prints results of training/test
# - x_test = attributes w/e final grade
# - y_test = actual grade
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Change MatPlotLib style
style.use("ggplot")

# Creates visual plot
# - p = x-value
# - labels our axes
# - .show() makes plot visible
p = "G1"
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel(predict)
pyplot.show()


