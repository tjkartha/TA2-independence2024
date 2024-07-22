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

# Create a DataFrame
data = {
    'Mean Squared Error (MSE)': [mse],
    'Root Mean Squared Error (RMSE)': [rmse],
    'Mean Absolute Error (MAE)': [mae],
    'R-squared (R2)': [r2]
}

df1 = pd.DataFrame(data, index=['YourModelName'])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import numpy as np

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy}')

# Precision
precision = precision_score(y_true, y_pred)
print(f'Precision: {precision}')

# Recall
recall = recall_score(y_true, y_pred)
print(f'Recall: {recall}')

# F1 Score
f1 = f1_score(y_true, y_pred)
print(f'F1 Score: {f1}')

# ROC AUC
roc_auc = roc_auc_score(y_true, y_pred_proba)
print(f'ROC AUC: {roc_auc}')

# Log Loss
log_loss_value = log_loss(y_true, y_pred_proba)
print(f'Log Loss: {log_loss_value}')


# Creating the DataFrame
data = {
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1_score],
    'ROC AUC': [roc_auc]
}

df2 = pd.DataFrame(data, index=['YourModelName'])
print(df2)