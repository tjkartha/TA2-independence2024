from models import *

# Get user's choice
def get_user_choice(num_options):
    while True:
        try:
            choice = int(input(f"Enter the number of the model you want to use (1-{num_options}): "))
            if 1 <= choice <= num_options:
                return choice
            else:
                print(f"Invalid choice. Please enter a number between 1 and {num_options}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def model_select(df):
    continue_splitting = True
    
    while continue_splitting:
        # Perform train-test split
        user_response = input("Do you want to perform a train-test split? (yes/no): ").strip().lower()
        if user_response == 'yes':
            target_column = input("Enter the name of the target column for the split: ").strip()
            if target_column not in df.columns:
                print(f"Column '{target_column}' not found in the dataset. Please try again.")
                continue

            test_size = float(input("Enter the test size (e.g., 0.2 for 20% test data): ").strip())
            X_train, X_test, y_train, y_test = perform_train_test_split(df, target_column, test_size)
            print("Train-test split performed.")
            print("X_train:\n", X_train.head())  # Print only the head for brevity
            print("X_test:\n", X_test.head())    # Print only the head for brevity
            print("y_train:\n", y_train.head())  # Print only the head for brevity
            print("y_test:\n", y_test.head())    # Print only the head for brevity
        else:
            print("Train-test split not performed.")
            return

        # Model selection
        models = [
            "Decision Tree Classifier",
            "Linear Regression",
            "Logistic Regression",
            "Random Forest Classifier",
            "Decision Tree Regressor",
            "Random Forest Regressor"
        ]
        display_models(models)
        choice = get_user_choice(len(models))
        selected_model = models[choice - 1]

        print(f"You selected: {selected_model}")

        # Train and evaluate the model
        if selected_model == "Decision Tree Classifier":
            model = train_decision_tree_classifier(X_train, y_train)
            evaluate_model_classifier(model, X_test, y_test)
        
        elif selected_model == "Linear Regression":
            model = train_linear_regression(X_train, y_train)
            print_performance_metrics_regressor(model, X_test, y_test)
    
        elif selected_model == "Decision Tree Regressor":
            model = train_decision_tree_regression(X_train, y_train)
            print_performance_metrics_regressor(model, X_test, y_test)
    
        elif selected_model == "Random Forest Regressor":
            model = train_random_forest_regression(X_train, y_train)
            print_performance_metrics_regressor(model, X_test, y_test)

        elif selected_model == "Logistic Regression":
            model = train_logistic_regression_classifier(X_train, y_train)
            evaluate_model_classifier(model, X_test, y_test)

        elif selected_model == "Random Forest Classifier":
            model = train_random_forest_classifier(X_train, y_train)
            evaluate_model_classifier(model, X_test, y_test)
        
        # Ask the user if they want to perform the split again
        repeat_response = input("Do you want to perform another train-test split? (yes/no): ").strip().lower()
        if repeat_response != 'yes':
            continue_splitting = False
