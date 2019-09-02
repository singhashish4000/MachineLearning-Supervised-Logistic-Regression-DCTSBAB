#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:44:08 2019

@author: ashish
"""

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

dataset = pd.read_csv('new_appdata10.csv')

# Data Preprocessing 

response = dataset["enrolled"]
dataset = dataset.drop(columns='enrolled')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response, test_size=0.2, random_state=0)

train_identifier = X_train['user']
X_train = X_train.drop(columns='user')

test_identifier = X_test['user']
X_test = X_test.drop(columns='user')


from sklearn.preprocessing import StandardScalar
sc_X = StandardScalar()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))





















