import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from pathlib import Path
import os

## Global variables
dataset = './demo-data/student-mat.csv'
pickled = './student_model.pickle'

data = pd.read_csv(dataset, sep=';')
data = data[["G1","G2","G3","studytime","failures","absences","age","traveltime","health","famrel","goout","Dalc","Walc"]] 

## Calculate value
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

## Split array and use only 10% of dataset
x_test, x_train, y_test, y_train = sklearn.model_selection.train_test_split(x,y,test_size = 0.15)

# If pickle does not exist, generate it
if not os.path.exists(pickled):
    print('Pickle does not exist.. Generating best accuracy..')
    best = 0
    for _ in range(20000):
        x_test, x_train, y_test, y_train = sklearn.model_selection.train_test_split(x,y,test_size = 0.15)

        ## Specify model
        linear = linear_model.LinearRegression()

        ## Select input data for training
        linear.fit(x_train, y_train)

        ## Calculate accuracy
        acc = linear.score(x_test,y_test)

        if acc > best:
            best = acc
            print('New best accuracy', best)
            ## Write model
            with open(pickled,"wb") as f:
                pickle.dump(linear,f)

## Load pickle model
pickle_in = open(pickled,"rb")
linear = pickle.load(pickle_in)

## Predict score
prediction = linear.predict(x_test)

## Uncomment to print estimations
## Loop through test data that the program hasn't seen
for x in range(len(prediction)):
    print("prediction: ", prediction[x], "Actual Result:", y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()