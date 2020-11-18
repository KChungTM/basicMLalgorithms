import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Data table of Car MPG
data = pd.read_csv("auto-mpg.csv")

# Separation of Attributes
data = data[["mpg", "displacement", "horsepower", "weight", "acceleration"]]

# Desired Prediction Value
predict = "mpg"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

accuracy = linear.score(x_test, y_test)
print(accuracy)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
