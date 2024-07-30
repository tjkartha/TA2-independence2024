
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
    models = ["Model A", "Model B", "Model C", "Model D","Model E"]
    model_codes = {
        "Model A": "print('Decision Tree')",
        "Model B": "print('Linear regression')",
        "Model C": "print('Logistic regrssion')",
        "Model D": "print('Random forest')",
        "Model E": "print('KNN')"
    }

    display_models(models)
    choice = get_user_choice(len(models))
    selected_model = models[choice - 1]
    selected_code = model_codes[selected_model]

    print(f"You selected: {selected_model}")
    print("Here is the code for the selected model:")
    print(selected_code)
