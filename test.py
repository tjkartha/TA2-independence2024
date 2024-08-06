# data = load_data("C:\Users\deepi\OneDrive\Documents\GitHub\TA2_Model Team\penguins.csv")
import os
import pandas as pd
from modelbuild import load_data, perform_train_test_split
from model_selection import display_models, get_user_choice
from DecisionTree_classifier import train_decision_tree_classifier, evaluate_model

def main():
    # Load the dataset
    data_path = pd.read_csv("C:/Users/deepi/OneDrive/Documents/GitHub/TA2_Model Team/penguins.csv")
    # data = load_data(data_path)

    # Perform train-test split
    user_response = input("Do you want to perform a train-test split on the target column 'sex'? (yes/no): ").strip().lower()
    if user_response == 'yes':
        test_size = float(input("Enter the test size (e.g., 0.2 for 20% test data): ").strip())
        X_train, X_test, y_train, y_test = perform_train_test_split(data_path, 'sex', test_size)
        print("Train-test split performed.")
        print("X_train:\n", X_train.head())  # Print only the head for brevity
        print("X_test:\n", X_test.head())    # Print only the head for brevity
        print("y_train:\n", y_train.head())  # Print only the head for brevity
        print("y_test:\n", y_test.head())    # Print only the head for brevity
    else:
        print("Train-test split not performed.")
        return

    # Model selection
    models = ["Model A", "Model B", "Model C", "Model D", "Model E", "Model F"]
    display_models(models)
    choice = get_user_choice(len(models))
    selected_model = models[choice - 1]

    print(f"You selected: {selected_model}")

    # Train and evaluate the model
    if selected_model == "Model A":  # Decision Tree classifier
        model = train_decision_tree_classifier(X_train, y_train)
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
        print("Decision Tree Classifier Performance:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", conf_matrix)

if __name__ == "_main_":
    main()