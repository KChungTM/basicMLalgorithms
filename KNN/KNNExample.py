import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

# Creates Label Encoder to change attributes to integers
le = preprocessing.LabelEncoder()

# Assigns integer values to categorical data
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# Value we want
predict = "class"

# Zip creates tuples and puts all the information
# into one large list
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size= 0.1)

# Creates classifier for data
model = KNeighborsClassifier(n_neighbors=9)

# Fits data with training data
model.fit(x_train, y_train)

# Finds accuracy of test set
acc = model.score(x_test,y_test)
print(acc)

# Predicted values using model and test set
predicted = model.predict(x_test)

# Names of the different classes
names = ["unacc", "acc", "good", "vgood"]

# Loop to print results
# - predicted[x] = Predicted value from model
# - x_test[x] = Data used in the model
# - y_test[x] = Actual value
# - names[] convert integer to categorical
for x in range(len(x_test)):
    print("Predicted:", names[predicted[x]], " Data: ", x_test[x], " Actual:", names[y_test[x]])
    # Returns neighbor's distances and indexes
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)