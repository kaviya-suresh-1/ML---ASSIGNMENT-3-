print("KAVIYA - 24BAD059")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(r"C:\Users\Kaviya\Desktop\Machine_learning_lab\auto-mpg (1).csv")
print(df.head())
df = df[['mpg', 'horsepower']]
df['horsepower'] = df['horsepower'].replace('?', np.nan)
df['horsepower'] = df['horsepower'].astype(float)
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
X = df[['horsepower']]
y = df['mpg']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
results = {}
degrees = [2, 3, 4]
for degree in degrees:
    print(f"\nPolynomial Degree: {degree}")
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    results[degree] = {
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "RMSE": test_rmse,
        "R2": test_r2
    }
    print(f"Train MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"RMSE: {test_rmse:.2f}")
    print(f"R² Score: {test_r2:.2f}")
print("\nRidge Regression (Degree = 4)")
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)
y_test_ridge = ridge.predict(X_test_poly)
ridge_mse = mean_squared_error(y_test, y_test_ridge)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, y_test_ridge)
print(f"Ridge MSE: {ridge_mse:.2f}")
print(f"Ridge RMSE: {ridge_rmse:.2f}")
print(f"Ridge R² Score: {ridge_r2:.2f}")
X_plot = np.sort(X.values, axis=0)
X_plot_scaled = scaler.transform(X_plot)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', alpha=0.5, label='Actual Data')
colors = ['red', 'blue', 'green']
for degree, color in zip(degrees, colors):
    poly = PolynomialFeatures(degree=degree)
    X_poly_plot = poly.fit_transform(X_plot_scaled)
    model = LinearRegression()
    model.fit(poly.fit_transform(X_train_scaled), y_train)
    y_plot = model.predict(X_poly_plot)   
    plt.plot(X_plot, y_plot, color=color, label=f'Degree {degree}')
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression Curve Fitting")
plt.legend()
plt.show()
plt.close()

train_errors = [results[d]['Train MSE'] for d in degrees]
test_errors = [results[d]['Test MSE'] for d in degrees]
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, test_errors, marker='o', label='Testing Error')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Testing Error")
plt.legend()
plt.show()
plt.close()

best_degree = 3
poly = PolynomialFeatures(degree=best_degree)
X_test_poly = poly.fit_transform(X_test_scaled)
model = LinearRegression()
model.fit(poly.fit_transform(X_train_scaled), y_train)
y_test_pred = model.predict(X_test_poly)
residuals = y_test - y_test_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.title("Residual Distribution (Degree 3)")
plt.show()
plt.close()
