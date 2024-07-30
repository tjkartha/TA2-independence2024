import pandas as pd
from sklearn.model_selection import train_test_split

# Function to perform train-test split
def perform_train_test_split(data, target_column, test_size=0.2):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to ask user for file input
def get_file_input():
    file_path = input("Please provide the path to your dataset file (CSV format): ").strip()
    try:
        data = pd.read_csv(file_path)
        print("File loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Function to display models
def display_models(models):
    print("Please select a model from the list below:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

# Function to get user's model choice
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

# Main function
def main():
    data = get_file_input()
    if data is None:
        print("Failed to load data. Exiting program.")
        return
    
    # Ask user if they want to perform train-test split
    user_response = input("Do you want to perform a train-test split on the target column? (yes/no): ").strip().lower()

    if user_response == 'yes':
        target_column = input("Enter the target column name: ").strip()
        test_size = float(input("Enter the test size (e.g., 0.2 for 20% test data): ").strip())
        X_train, X_test, y_train, y_test = perform_train_test_split(data, target_column, test_size)
        print("Train-test split performed.")
        print("X_train:\n", X_train.head())
        print("X_test:\n", X_test.head())
        print("y_train:\n", y_train.head())
        print("y_test:\n", y_test.head())
    else:
        print("Train-test split not performed.")

    models = ["Decision Tree", "Linear Regression", "Logistic Regression", "Random Forest", "KNN"]
    model_codes = {
        "Decision Tree": "print('Decision Tree')",
        "Linear Regression": "print('Linear Regression')",
        "Logistic Regression": "print('Logistic Regression')",
        "Random Forest": "print('Random Forest')",
        "KNN": "print('KNN')"
    }

    display_models(models)
    choice = get_user_choice(len(models))
    selected_model = models[choice - 1]
    selected_code = model_codes[selected_model]

    print(f"You selected: {selected_model}")
    print("Here is the code for the selected model:")
    print(selected_code)

if __name__ == "__main__":
    main()
