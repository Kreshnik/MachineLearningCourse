import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import datetime
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class StockPredictor:

    def __init__(self):
        self.linearRegressionClassifier = linear_model.LinearRegression()
        self.polynomialClassifier = make_pipeline(PolynomialFeatures(2), Ridge())
        self.svrClassifier = SVR(kernel='linear', C=1e3)

        self.dataFrame = pd.read_csv('AAPL.csv')
        self.dataFrame['Year'] = pd.DatetimeIndex(self.dataFrame['Date']).year.astype(int)
        self.dataFrame['Month'] = pd.DatetimeIndex(self.dataFrame['Date']).month
        self.dataFrame['Day'] = pd.DatetimeIndex(self.dataFrame['Date']).day

    def train(self, period="Year"):
        self.linearRegressionClassifier.fit(
            self.dataFrame[period].values.reshape(-1, 1),
            self.dataFrame['Adj Close'].values.reshape(-1, 1)
        )
        self.polynomialClassifier.fit(
            self.dataFrame[period].values.reshape(-1, 1),
            self.dataFrame['Adj Close'].values.reshape(-1, 1)
        )
        self.svrClassifier.fit(
            self.dataFrame[period].values.reshape(-1, 1),
            self.dataFrame['Adj Close'].values.reshape(-1, 1)
        )

    def predict(self, numeric_value=0):
        return {
            'linear': self.linearRegressionClassifier.predict(numeric_value)[0][0],
            'polynomial': self.polynomialClassifier.predict(numeric_value)[0][0],
            'svr': self.svrClassifier.predict(numeric_value)[0]
        }

    def plot_prediction(self, period="Year", numeric_value=0):
        style.use('ggplot')
        mpl.rc('figure', figsize=(8, 7))
        predictions = self.predict(numeric_value)
        plt.scatter(self.dataFrame[period], self.dataFrame['Adj Close'], color='blue', label='Real')
        plt.scatter(numeric_value, predictions['linear'], color='red', label='Linear regression prediction')
        plt.scatter(numeric_value, predictions['polynomial'], color='green', label='Polynomial regression prediction')
        plt.scatter(numeric_value, predictions['svr'], color='orange', label='SVR regression prediction')
        plt.xlabel(period)
        plt.ylabel('Price')
        plt.title(period + ' Stock Prediction')
        plt.legend()
        plt.show()
