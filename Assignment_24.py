# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 03:26:59 2018

@author: Zakir
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

#Import datasets
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target
features["PRICE"] = boston.target

##Assigning the Dependant and independant variables
#Y = Price
#X = All the other attributes

X = features.drop('PRICE', axis = 1)
y = features['PRICE']

#Splitting dataset into tranign set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=15, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Plotting the scatter plots
plt.scatter(y_test, y_pred, color = ['red', 'green'])
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual Prices vs Predicted prices")
