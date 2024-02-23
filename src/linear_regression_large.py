#!/usr/bin/python
import pandas
from sklearn import linear_model

df = pandas.read_csv("data_files/linear_regression/winequality_scaled.csv")

X = df[["fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"]]
y = df["quality"]

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)
