import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from pathlib import Path
import os

## Global variables
dataset = datasets.load_breast_cancer
pickled = './.pickle'
