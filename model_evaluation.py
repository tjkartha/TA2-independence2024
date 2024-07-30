from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae}')

# Mean Squared Error
mse = mean_squared_error(y_true, y_pred)
print(f'MSE: {mse}')

# Root Mean Squared Error
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

# R-squared
r2 = r2_score(y_true, y_pred)
print(f'R²: {r2}')

# Adjusted R-squared
n = len(y_true)  # Number of samples
p = 1  # Number of predictors (adjust this based on your model)
adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
print(f'Adjusted R²: {adj_r2}')


