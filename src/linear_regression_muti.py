#!/usr/bin/python
import pandas
from sklearn import linear_model

df = pandas.read_csv("data_files/ex1data2_scaled.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)
