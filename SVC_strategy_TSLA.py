import yfinance as yf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-notebook')

import warnings
warnings.filterwarnings('ignore')

df = yf.download("TSLA", start='2021-01-03', end='2023-11-03')
df.index = pd.to_datetime(df.index)


df['Open-Close'] = df.Open-df.Close
df['High-Low'] = df.High-df.Low

X = df[['Open-Close', 'High-Low']]

print(df.head())

y = np.where(df['Close'].shift(1)>df['Close'], 1, 0)

split_percentage = 0.8
split = int(split_percentage*len(df))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

cls = SVC().fit(X_train, y_train)
y_predict = cls.predict(X_test)

accuracy_test = accuracy_score(y_test, y_predict)

print('Strategy Accuracy:{:.2f}%'.format(accuracy_test*100))

df['Predicted_Signal'] = cls.predict(X)

df['Returns'] = df.Close.pct_change()

df['Strategy_Returns'] = df.Returns * df.Predicted_Signal.shift(1)

df['Cumulative_Returns'] = (df.Strategy_Returns.iloc[split:int(len(df)*0.95)]+1).cumprod()

plt.title('Cumulative Returns Plot', fontsize=16)
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')

df['Cumulative_Returns'].plot(figsize=(15,7), color='g')
plt.show()



