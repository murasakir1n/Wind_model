import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('wind_dataset.csv')

print(df)

df = df.fillna(df.mean(numeric_only=True))

df['DATE'] = pd.to_datetime(df['DATE'])
df['month'] = df['DATE'].dt.month
df['year'] = df['DATE'].dt.year
df['day'] = df['DATE'].dt.day

df = pd.get_dummies(df, columns=['IND', 'IND.1', 'IND.2'])

X = df[['day', 'month' ,'year', 'IND','RAIN','IND.1', 'T.MAX', 'IND.2', 'T.MIN', 'T.MIN.G']].values
y = df['WIND'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_mean = X_train.mean(axis = 0)
train_std = X_train.std(axis = 0)

X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std

def predict(X, w, b):
    return X @ w + b

def gradient(X, y, w, b):
    n = len(y)
    y_pred = predict(X, w, b)
    error = y_pred - y
    dw = (2/n) * (X.T @ error)
    db = (2/n) * np.sum(error)
    return dw, db

w = np.zeros((X_train.shape[1], 1))
b = 0
lr = 0.01
steps = 10000

losses = []
for step in range(steps):
    dw, db = gradient(X_train_norm, y_train, w, b)
    w = w - lr * dw
    b = b - lr * db

    y_pred_train = predict(X_train_norm, w, b)
    mse = np.mean((y_pred_train - y_train) ** 2)
    losses.append(mse)

    if step % 100 == 0:
        print(f'Iteration {step}: MSE = {mse:.4f}, w = {np.round(w.flatten(), 4)}, b = {b:.4f}')


y_pred_test = predict(X_test_norm, w, b)
mse_test = np.mean((y_pred_test - y_test) ** 2)

print(f'\nФинальные параметры: w: {np.round(w.flatten(), 4)}, b: {b:.4f}')
print(f'MSE на тесте: {mse_test:.4f}')

















