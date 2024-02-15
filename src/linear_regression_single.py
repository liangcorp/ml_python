#!/usr/bin/python
import pandas
from sklearn import linear_model

df = pandas.read_csv("data_files/ex1data1.csv")

X = df[['Feature']]
y = df['Result']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)
