#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:10:00 2019

@author: kanav
"""
""" Cmd + i to inspect about some feature"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #Take all the columns except last one
y = dataset.iloc[:, -1].values #Take the last column as the result
#X-> details of the people (Feature), Y-> Result (Label)

#__________________________________________________________________________
# Taking care of missing data

#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#jaha jaha val nhi hai vaha average laga dia
imputer = Imputer(missing_values=np.nan, strategy='mean')
#Imputer is 2 step transformation -> Fit and Transform 

#???What is happening? 
imputer = imputer.fit(X.iloc[:, 1:]) #Imputer is fitted on Age and Salary
#Upar jo imputer function banaya tha jo sari nan values ko replace kardega mean se vo humne age and salary par lagadia
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:]) #??? in dono line me difference kya hai


#__________________________________________________________________________
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()
X.iloc[:, 0] = labelEncoder_x.fit_transform(X.iloc[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)


#__________________________________________________________________________

# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#__________________________________________________________________________
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc_X.transform(X_test[:, 3:])





















#__________________________________________________________________________