import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE 

warnings.filterwarnings('ignore')

data = yf.download('NVDA', start='2021-01-04', end='2024-05-04')
data.index = pd.to_datetime(data.index)

data['Open-Close'] = (data['Open'] - data['Close'])
data['High-Low'] = (data['High'] - data['Low'])
data['SMA'] = data['Adj Close'].rolling(window=10).mean()
data.dropna(inplace=True)

X = data[['Open-Close', 'High-Low', 'SMA']]
y = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, -1)

split = int(len(data)*0.75)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [None, 10,20,30],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4],
    'bootstrap': [True, False]

    }

grid_search = GridSearchCV(RandomForestClassifier(random_state = 5), param_grid, cv=5,
                           n_jobs =-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)


best_clf = grid_search.best_estimator_
print('Best parameters found: ', grid_search.best_params_)

y_pred = best_clf.predict(X_test)
print('Accuracy of Tuned Model Predictions: ', accuracy_score(y_test, y_pred)* 100.0)
print(classification_report(y_test, y_pred))
print('AUC-ROC', roc_auc_score(y_test, y_pred))
