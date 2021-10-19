import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pyplot
from sklearn import metrics
import pickle
from matplotlib import style
from pathlib import Path
import os

## Global variables
dataset = datasets.load_breast_cancer()
pickled = './.pickle'

x = dataset.data
y = dataset.target

## Split array and use only 10% of dataset
x_test, x_train, y_test, y_train = sklearn.model_selection.train_test_split(x,y,test_size = 0.15)

names = ["malignant","benign"]

clf = svm.SVC(kernel="linear", C=5)

clf.fit(x_train,y_train)

y_prediction = clf.predict (x_test)

acc = metrics.accuracy_score(y_test, y_prediction)

print(acc)