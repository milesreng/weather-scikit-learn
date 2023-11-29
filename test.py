import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import style
import matplotlib.pyplot as pyplot
import pickle

data = pd.read_csv("data/student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

# num_trials = 1000

# max_accuracy = 0
# best_model = None

# for _ in range(num_trials):

#   x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

#   linear = linear_model.LinearRegression()

#   linear.fit(x_train, y_train)

#   accuracy = linear.score(x_test, y_test)

#   print(accuracy)

#   if accuracy > max_accuracy:
#     max_accuracy = accuracy
#     best_model = linear

# print(max_accuracy)
# # save model
# with open("data/studentmodel.pickle", "wb") as f:
#   pickle.dump(best_model, f)

pickle_in = open("data/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

accuracy = linear.score(x_test, y_test)
print("accuracy: " + str(accuracy))

predictions = linear.predict(x_test)

for x in range(len(predictions)):
  print(predictions[x], x_test[x], y_test[x])

print("coefficient: " + str(linear.coef_))
print("intercept: " + str(linear.intercept_))

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()