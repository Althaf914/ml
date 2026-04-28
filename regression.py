import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = load_diabetes()
X = data.data
y = data.target

X_simple = X[:, [0]]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)

print("Simple Linear Regression")
print(mean_squared_error(y_test_s, y_pred_s))
print(r2_score(y_test_s, y_pred_s))
print(model_simple.coef_)
print(model_simple.intercept_)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y, test_size=0.2, random_state=42)

model_multiple = LinearRegression()
model_multiple.fit(X_train_m, y_train_m)
y_pred_m = model_multiple.predict(X_test_m)

print("Multiple Linear Regression")
print(mean_squared_error(y_test_m, y_pred_m))
print(r2_score(y_test_m, y_pred_m))
print(model_multiple.coef_)
print(model_multiple.intercept_)
