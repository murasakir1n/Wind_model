import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Загружаем Iris дата сет
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)


X = df[['petal width (cm)', 'sepal length (cm)', 'sepal width (cm)']].values
y = df[['petal length (cm)']].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация
mean_train = X_train.mean(axis = 0)
std_train = X_train.std(axis = 0)

X_train_norm = (X_train - mean_train) / std_train
X_test_norm = (X_test - mean_train) / std_train

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
lr = 0.05
steps = 1000

losses = []
for i in range(steps):
    dw, db = gradient(X_train_norm, y_train,w, b)
    w = w - lr * dw
    b = b - lr * db

    y_pred_train = predict(X_train_norm, w, b)
    mse = np.mean((y_pred_train - y_train) ** 2)
    losses.append(mse)

    if i % 100 == 0:
        print(f'Итерация {i}: MSE = {mse:.4f} w = {w.flatten()}, b = {b:.4f}')

y_pred_test = predict(X_test_norm, w, b)
test_mse = np.mean((y_pred_test - y_test) ** 2)
print(f"\nФинальные параметры: w = {w.flatten()[0]:.4f}, b = {b:.4f}")
print(f"MSE на тесте: {test_mse:.4f}")

import matplotlib.pyplot as plt

# Предсказания на train и test

# График: реальные vs предсказанные
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred_train, color='blue', label='Train', alpha=0.6)
plt.scatter(y_test, y_pred_test, color='green', label='Test', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Идеальная линия (y = x)')
plt.xlabel('Реальная длина лепестка (cm)')
plt.ylabel('Предсказанная длина лепестка (cm)')
plt.title('Реальные vs Предсказанные значения')
plt.legend()
plt.grid(True)
plt.show()


# scaler = StandardScaler()
# X_train_sc = scaler.fit_transform(X_train)
# X_test_sc = scaler.transform(X_test)
#
# model = LinearRegression()
# model.fit(X_train_sc, y_train)
# print("sklearn w:", model.coef_.flatten(), "b:", model.intercept_)
# print("sklearn test MSE:", np.mean((model.predict(X_test_sc) - y_test)**2))




