import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

plt.style.use('tableau-colorblind10')

data = yf.download('V', start='2022-05-04', end='2023-05-04')['Adj Close']

print(data.head())

data.plot(figsize=(15,7))
plt.title('Visa Security')
plt.ylabel('Dollar Price')
plt.legend(['Adj_Close'])
plt.show()

from statsmodels.tsa.stattools import adfuller

p_value = adfuller(data)[1]

if p_value > 0.05:
    print('The p value is {}, so time series is not stationary'.format(p_value))

else:
    print('Time series is stationary')

from statsmodels.graphics.tsaplots import plot_acf

plt.rcParams['figure.figsize']=(16,8)
plot_acf(data)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation of Close Prices')
plt.show()

data.diff().plot(figsize=(17,7))
plt.xlabel('Date')
plt.legend(['First Order Differences'])
plt.show()

sns.displot(data.diff().dropna(), kde=True, aspect=10/7)
plt.show()

split = int(len(data)*0.87)
train_set, test_set = data[:split], data[split:]

plt.figure(figsize=(16,8))
plt.title('Visa Price')
plt.xlabel('Date')
plt.ylabel('Dollar Price')
plt.plot(train_set, 'green')
plt.plot(test_set, 'red')
plt.legend(['Training Set', 'Testing Data'])
plt.show()

import warnings
warnings.filterwarnings('ignore')

from statsmodels.graphics.tsaplots import plot_pacf

plt.rcParams['figure.figsize']=(16,8)
plot_pacf(train_set.diff().dropna(), lags=20)
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation')
plt.show()

plt.rcParams['figure.figsize'] = (16,8)
plot_acf(train_set.diff().dropna(), lags=20)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.show()

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA

aic_p = []
bic_p = []

p = range(1, 6)

for i in p:
    model = ARIMA(train_set, order=(i, 1, 0))
    model_fit = model.fit()

    aic_temp = model_fit.aic
    bic_temp = model_fit.bic

    aic_p.append(aic_temp)
    bic_p.append(bic_temp)


plt.figure(figsize=(16,8))
plt.plot(range(1,6), aic_p, color='red')
plt.plot(range(1,6), bic_p, color='purple')
plt.title('Assessing AR term')
plt.xlabel('P for AR term')
plt.ylabel('AIC & BIC Scores')
plt.legend(['AIC Score', 'BIC Scores'])
plt.show()

aic_q = []
bic_q = []

q = range(1,6)

for i in q:
    model = ARIMA(train_set, order=(0,1,i))
    model_fit = model.fit()

    aic_temp = model_fit.aic
    bic_temp = model_fit.bic

    aic_q.append(aic_temp)
    bic_q.append(bic_temp)

plt.figure(figsize=(16,8))
plt.plot(range(1,6), aic_q, color='red')
plt.plot(range(1,6), bic_q, color='purple')

plt.title('Assessing MA term')
plt.xlabel('Q for MA term')
plt.ylabel('AIC & BIC Scores')
plt.legend(['AIC Score', 'BIC Score'])
plt.show()

model = ARIMA(train_set, order=(1, 1, 1))
model_fit_0 = model.fit()

past = train_set.tolist()
predictions= []

test_set = test_set[:50]

for i in range(len(test_set)):

    model = ARIMA(past, order = (1,1,1))
    model_fit = model.fit(start_params = model_fit_0.params)

    forecast_results = model_fit.forecast()
    
    pred = forecast_results[0]
    predictions.append(pred)

    past.append(test_set[i])

for i in range(0,10):
    print('Predicted Value = {pred}, Actual Value = {test}'.format(
        pred=predictions[i], test=test_set[i]))


residual = []

for i in range(len(test_set)):
    residual.append(predictions[i]-test_set[i])



plt.figure(figsize=(15,7))
plt.plot(residual)
plt.show()



plt.figure(figsize=(15,7))
sns.displot(residual, kde=True, height=6, aspect=11/6)
plt.show()
    

from sklearn.metrics import mean_squared_error

error = mean_squared_error(test_set, predictions)
print('Test MSE: {mse}'.format(mse=error))


plt.figure(figsize=(16,8))
plt.plot(test_set)
plt.plot(test_set.index, predictions, color='red')
plt.title('Visa Stock')
plt.xlabel('Trading Date')
plt.ylabel('Price')
plt.legend(['test_set', 'predictions'])
plt.show()
    




