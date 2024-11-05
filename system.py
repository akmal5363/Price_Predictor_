import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Признаки домов (площадь и количество комнат)
X = np.array([[100, 3], [80, 2], [120, 4], [90, 3], [70, 2],
              [110, 3], [85, 2], [95, 3], [105, 3], [75, 2]])

# Целевая переменная (цены домов)
y = np.array([500000, 400000, 600000, 450000, 350000,
              520000, 420000, 480000, 510000, 370000])

# Шаг 1: Создание контрольной выборки (10% от данных)
X_temp, X_control, y_temp, y_control = train_test_split(X, y, test_size=0.1, random_state=42)

# Шаг 2: Разделение оставшихся данных (90%) на обучающую и валидационную выборки (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Создаем и обучаем модель на обучающей выборке
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем предсказания на валидационной выборке
y_val_pred = model.predict(X_val)

# Оцениваем модель на валидационной выборке с использованием MSE
val_mse = mean_squared_error(y_val, y_val_pred)
print(f"Mean Squared Error на валидационной выборке: {val_mse}")

# Делаем предсказания на контрольной выборке
y_control_pred = model.predict(X_control)

# Оцениваем модель на контрольной выборке с использованием MSE
control_mse = mean_squared_error(y_control, y_control_pred)
print(f"Mean Squared Error на контрольной выборке: {control_mse}")
