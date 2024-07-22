# print("model-eval")
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to perform train-test split
def perform_train_test_split(data, target_column, test_size=0.2):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Ask user if they want to perform train-test split
user_response = input("Do you want to perform a train-test split on the target column? (yes/no): ").strip().lower()

if user_response == 'yes':
    target_column = 'target'  # You can modify this to prompt user for the target column
    test_size = float(input("Enter the test size (e.g., 0.2 for 20% test data): ").strip())
    X_train, X_test, y_train, y_test = perform_train_test_split(data, target_column, test_size)
    print("Train-test split performed.")
    print("X_train:\n", X_train)
    print("X_test:\n", X_test)
    print("y_train:\n", y_train)
    print("y_test:\n", y_test)
else:
    print("Train-test split not performed.")



def display_models(models):
    print("Please select a model from the list below:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

def get_user_choice(num_models):
    while True:
        try:
            choice = int(input("Enter the number of the model you want to select: "))
            if 1 <= choice <= num_models:
                return choice
            else:
                print(f"Please enter a number between 1 and {num_models}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    models = ["Model A", "Model B", "Model C", "Model D","Model E","Model F","Model G"]
    model_codes = {
        "Model A": "print('Decision Tree')",
        "Model B": "print('Linear regression')",
        "Model C": "print('Logistic regrssion')",
        "Model D": "print('Random forest')",
        "Model E": "print('KNN')",
        "Model F": "print('Naive Byes')",
        "Model G": "print('k-mean')"
    }

    display_models(models)
    choice = get_user_choice(len(models))
    selected_model = models[choice - 1]
    selected_code = model_codes[selected_model]

    print(f"You selected: {selected_model}")
    print("Here is the code for the selected model:")
    print(selected_code)


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