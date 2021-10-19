import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model,preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from pathlib import Path
import os

## Global variables
dataset = './demo-data/car.data'
pickled = './car_model.pickle'

data = pd.read_csv(dataset)

## Read non-numerical values into numerical values
encoder = preprocessing.LabelEncoder()

## Assign values by column
buying = encoder.fit_transform(list(data["buying"]))
maint = encoder.fit_transform(list(data["maint"]))
door = encoder.fit_transform(list(data["door"]))
persons = encoder.fit_transform(list(data["persons"]))
lug_boot = encoder.fit_transform(list(data["lug_boot"]))
safety = encoder.fit_transform(list(data["safety"]))
carClass = encoder.fit_transform(list(data["class"]))

predict = "class"

## Zip will combine into a single list, y defines the class that will be estimated
x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(carClass)

## Split array and use only 10% of dataset
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.15)

## How many neighbours to use
knn = KNeighborsClassifier(9)

# If pickle does not exist, generate it
if not os.path.exists(pickled):
    print('Pickle does not exist.. Generating best accuracy..')
    best = 0
    for _ in range(10):
        ## How many neighbours to use
        knn = KNeighborsClassifier(9)

        ## Train the model
        knn.fit(x_train,y_train)

        ## Print accuracy
        acc = knn.score(x_test,y_test)

        if acc > best:
            best = acc
            print('New best accuracy', best)
            with open(pickled,"wb") as f:
                pickle.dump(knn,f)

## Load pickle model
pickle_in = open(pickled,"rb")
knn = pickle.load(pickle_in)

prediction = knn.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

correct = 0
incorrect = 0

for car in range(len(prediction)):
    if names[prediction[car]] == names[y_test[car]]:
        correct = correct +1
    else:
        incorrect = incorrect +1

print("Total Correct: ", correct)
print("Total Incorrect: ", incorrect)
