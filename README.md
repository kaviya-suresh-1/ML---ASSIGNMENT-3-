Scenario 1 – Multilinear Regression: Student Performance
Using the Students Performance in Exams dataset (Kaggle), we predict the Final Exam Score (average of Math, Reading, Writing) from factors like study hours, attendance, parental education, test prep, and sleep hours.
Steps:
- Import libraries, load data, preprocess & encode features
- Handle missing values, scale inputs, split train/test
- Train a Multilinear Regression model
- Evaluate with MSE, RMSE, R²
- Interpret coefficients, optimize with feature elimination, Ridge, Lasso
Visuals: Predicted vs Actual scores, coefficient magnitudes, residual plots

 Scenario 2 – Polynomial Regression: Vehicle Fuel Efficiency
Using the Auto MPG dataset (Kaggle), we predict Miles Per Gallon (MPG) from engine horsepower, modeling the non-linear relationship.
Steps:
- Import libraries, clean data, handle missing values
- Generate polynomial features (degrees 2–4), scale inputs
- Train/test split, fit Polynomial Regression models
- Evaluate with MSE, RMSE, R²
- Compare degrees, apply Ridge regression to reduce overfitting
Visuals: Polynomial curve fits, training vs testing error, over/underfitting demo

 Outcomes
- Compare linear vs non-linear regression
- Apply preprocessing, scaling, and imputation
- Evaluate models with metrics and visualizations
- Use regularization for robust predictions
