#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 09:48:52 2019

@author: ashish
"""


### Importing Libraries ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from dateutil import parser

dataset = pd.read_csv('data/appdata10/appdata10.csv')

### EDA ###
dataset.head()
dataset.describe()


### Data Cleaning ###
dataset['hour'] = dataset.hour.str.slice(1, 3).astype(int)

### Ploting ###

temp_dataset = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])

temp_dataset.head()

### Histograms ###

plt.suptitle('Histograms of Numerical Columns', fontsize = 20)

for i in range(1, temp_dataset.shape[1] + 1):
    plt.subplot(3, 3 ,i)
    f = plt.gca()
    f.set_title(temp_dataset.columns.values[i -1])
    
    vals = np.size(temp_dataset.iloc[:, i -1].unique())
    
    plt.hist(temp_dataset.iloc[:, i -1], bins = vals, color = ['#3F5D7D'])


### Corealtion with response ###

temp_dataset.corrwith(dataset.enrolled).plot.bar(figsize=(20, 10), title = 'Correaltion with Response Variable', fontsize = 15, rot = 45, grid=True)



### Correaltion Matrix ###

sn.set(style="white", font_scale=2)

# Compute the correaltion matrix
corr = temp_dataset.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(7, 10))
f.suptitle('Correlation Matrix', fontsize = 10)

# Generate a custom diverging colormap

cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})













































































