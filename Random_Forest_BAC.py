import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf

data = yf.download('BAC', start='2021-01-04', end='2024-05-04')
data.index = pd.to_datetime(data.index)

data['Open-Close'] = (data['Open'] - data['Close'])
data['High-Low'] = (data['High'] - data['Low'])

data.dropna(inplace=True)

X = data[['Open-Close', 'High-Low']]

y = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, -1)

split = int(len(data)*0.75)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

clf = RandomForestClassifier(random_state=5)
BAC_model = clf.fit(X_train, y_train)

print('Accuracy of Model Predictions: ', accuracy_score(y_test,
                                                        BAC_model.predict(X_test),
                                                        normalize=True)*100.0)
