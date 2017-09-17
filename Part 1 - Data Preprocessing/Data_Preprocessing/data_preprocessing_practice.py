#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

#Load dataset
dataset = pd.read_csv('Data.csv')

#Create independent matrix
X = dataset.iloc[:, :-1].values

#Create dependent matrix
Y = dataset.iloc[:, 3].values

#Fix missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Categorical country column
#Encode 
labelEncoder = LabelEncoder()
X[:, 0] = labelEncoder.fit_transform(X[:, 0])
Y = labelEncoder.fit_transform(Y)

oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()