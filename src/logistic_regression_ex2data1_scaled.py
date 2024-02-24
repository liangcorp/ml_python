#!/usr/bin/python
import pandas
from sklearn.linear_model import LogisticRegression

df = pandas.read_csv("data_files/logistic_regression/ex2data1_scaled.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

logisticRegr = LogisticRegression()
logisticRegr.fit(X, y)

print(logisticRegr.coef_)
