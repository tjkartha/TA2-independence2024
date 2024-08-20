# import os
# from model_build import perform_train_test_split
# from model_selection import display_models, get_user_choice
# from decision_tree_classi import train_decision_tree_classifier, evaluate_model
# from decision_tree_reg import train_decision_tree_regression, print_performance_metrics_regressor
# from linear_reg import train_linear_regression, print_performance_metrics_regressor
# from logistic_reg import train_logistic_regression_classifier, print_performance_metrics_classifier
# from random_forest_c import train_random_forest_classifier, evaluate_model
# from Random_forest_r import train_random_forest_regression, print_performance_metrics_regressor

# # Import the dataset from test.py
# from test import data

# def main():
#     continue_splitting = True
    
#     while continue_splitting:
#         # Perform train-test split
#         user_response = input("Do you want to perform a train-test split on the target column 'sex'? (yes/no): ").strip().lower()
#         if user_response == 'yes':
#             test_size = float(input("Enter the test size (e.g., 0.2 for 20% test data): ").strip())
#             X_train, X_test, y_train, y_test = perform_train_test_split(data, 'sex', test_size)
#             print("Train-test split performed.")
#             print("X_train:\n", X_train.head())  # Print only the head for brevity
#             print("X_test:\n", X_test.head())    # Print only the head for brevity
#             print("y_train:\n", y_train.head())  # Print only the head for brevity
#             print("y_test:\n", y_test.head())    # Print only the head for brevity
#         else:
#             print("Train-test split not performed.")
#             return

#         # Model selection
#         models = ["Decision Tree Classifier", "Linear Regression", "Logistic Regression", "Random Forest Classifier", "Decision Tree Regressor", "Random Forest Regressor"]
#         display_models(models)
#         choice = get_user_choice(len(models))
#         selected_model = models[choice - 1]

#         print(f"You selected: {selected_model}")

#         # Train and evaluate the model
#         if selected_model == "Decision Tree Classifier":
#             model = train_decision_tree_classifier(X_train, y_train)
#             accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
#             print("Decision Tree Classifier Performance:")
#             print("Accuracy:", accuracy)
#             print("Precision:", precision)
#             print("Recall:", recall)
#             print("F1 Score:", f1)
#             print("Confusion Matrix:\n", conf_matrix)
        
#         elif selected_model == "Linear Regression":
#             model = train_linear_regression(X_train, y_train)
#             y_pred = model.predict(X_test)
#             print("Linear Regression Performance:")
#             print_performance_metrics_regressor(y_test, y_pred, 'Linear Regression')
        
#         elif selected_model == "Logistic Regression":
#             model = train_logistic_regression_classifier(X_train, y_train)
#             y_pred = model.predict(X_test)
#             print("Logistic Regression Performance:")
#             print_performance_metrics_classifier(y_test, y_pred, 'Logistic Regression')
        
#         elif selected_model == "Random Forest Classifier":
#             model = train_random_forest_classifier(X_train, y_train)
#             accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
#             print("Random Forest Classifier Performance:")
#             print("Accuracy:", accuracy)
#             print("Precision:", precision)
#             print("Recall:", recall)
#             print("F1 Score:", f1)
#             print("Confusion Matrix:\n", conf_matrix)
        
#         elif selected_model == "Decision Tree Regressor":
#             model = train_decision_tree_regression(X_train, y_train)
#             y_pred = model.predict(X_test)
#             print("Decision Tree Regressor Performance:")
#             print_performance_metrics_regressor(y_test, y_pred, 'Decision Tree Regressor')
        
#         elif selected_model == "Random Forest Regressor":
#             model = train_random_forest_regression(X_train, y_train)
#             y_pred = model.predict(X_test)
#             print("Random Forest Regressor Performance:")
#             print_performance_metrics_regressor(y_test, y_pred, 'Random Forest Regressor')
        
#         # Ask the user if they want to perform the split again
#         repeat_response = input("Do you want to perform another train-test split? (yes/no): ").strip().lower()
#         if repeat_response != 'yes':
#             continue_splitting = False

# if __name__ == "_main_":
#     main()




import os
from model_build import load_data, perform_train_test_split
from model_selection import display_models, get_user_choice
from decision_tree_classi import train_decision_tree_classifier, evaluate_model
from decision_tree_reg import train_decision_tree_regression, print_performance_metrics_regressor
from linear_reg import train_linear_regression, print_performance_metrics_regressor
from logistic_reg import train_logistic_regression_classifier, print_performance_metrics_classifier
from random_forest_c import train_random_forest_classifier, evaluate_model
from Random_forest_r import train_random_forest_regression, print_performance_metrics_regressor
# from evaluation import print_performance_metrics_regressor
def main(data):
    continue_splitting = True
    
    while continue_splitting:
        # Perform train-test split
        user_response = input("Do you want to perform a train-test split on the target column 'sex'? (yes/no): ").strip().lower()
        if user_response == 'yes':
            test_size = float(input("Enter the test size (e.g., 0.2 for 20% test data): ").strip())
            X_train, X_test, y_train, y_test = perform_train_test_split(data, 'HeartDisease', test_size)
            print("Train-test split performed.")
            print("X_train:\n", X_train.head())  # Print only the head for brevity
            print("X_test:\n", X_test.head())    # Print only the head for brevity
            print("y_train:\n", y_train.head())  # Print only the head for brevity
            print("y_test:\n", y_test.head())    # Print only the head for brevity
        else:
            print("Train-test split not performed.")
            return

        # Model selection
        models = ["Decision Tree Classifier", "Linear Regression", "Logistic Regression", "Random Forest Classifier", "Decision Tree Regressor", "Random Forest Regressor"]
        display_models(models)
        choice = get_user_choice(len(models))
        selected_model = models[choice - 1]

        print(f"You selected: {selected_model}")

        # Train and evaluate the model
        if selected_model == "Decision Tree Classifier":
            model = train_decision_tree_classifier(X_train, y_train)
            accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
            print("Decision Tree Classifier Performance:")
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("Confusion Matrix:\n", conf_matrix)
        
        elif selected_model == "Linear Regression":
            model = train_linear_regression(X_train, y_train)
            accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
            print("Decision Tree Classifier Performance:")
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("Confusion Matrix:\n", conf_matrix)

        elif selected_model == "Linear Regression":
          model = train_linear_regression(X_train, y_train)
          y_pred = model.predict(X_test)
          print("Linear Regression Performance:")
          print_performance_metrics_regressor(y_test, y_pred, 'Linear Regression')
    
        elif selected_model == "Decision Tree Regressor":
           model = train_decision_tree_regression(X_train, y_train)
           y_pred = model.predict(X_test)
           print("Decision Tree Regressor Performance:")
           print_performance_metrics_regressor(y_test, y_pred, 'Decision Tree Regressor')
    
        elif selected_model == "Random Forest Regressor":
           model = train_random_forest_regression(X_train, y_train)
           y_pred = model.predict(X_test)
           print("Random Forest Regressor Performance:")
           print_performance_metrics_regressor(y_test, y_pred, 'Random Forest Regressor')


        elif selected_model == "Logistic Regression":
            model = train_logistic_regression_classifier(X_train, y_train)
            y_pred = model.predict(X_test)
            print("Logistic Regression Performance:")
            print_performance_metrics_classifier(y_test, y_pred, 'Logistic Regression')

        
        elif selected_model == "Random Forest Classifier":
            model = train_random_forest_classifier(X_train, y_train)
            accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
            print("Random Forest Classifier Performance:")
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("Confusion Matrix:\n", conf_matrix)
        
        # elif selected_model == "Decision Tree Regressor":
        #     model = train_decision_tree_regression(X_train, y_train)
        #     print("Decision Tree Regressor Performance:")
        #     print_performance_metrics_regressor(model, X_test, y_test)
        
        # elif selected_model == "Random Forest Regressor":
        #     model = train_random_forest_regression(X_train, y_train)
        #     print("Random Forest Regressor Performance:")
        #     print_performance_metrics_regressor(model, X_test, y_test)
        
        # Ask the user if they want to perform the split again
        repeat_response = input("Do you want to perform another train-test split? (yes/no): ").strip().lower()
        if repeat_response != 'yes':
            continue_splitting = False

if __name__== "_main_":
    main()


# import os
# from model_build import perform_train_test_split
# from model_selection import display_models, get_user_choice
# from decision_tree_classi import train_decision_tree_classifier, evaluate_model
# from decision_tree_reg import train_decision_tree_regression, print_performance_metrics_regressor
# from linear_reg import train_linear_regression, print_performance_metrics_regressor
# from logistic_reg import train_logistic_regression_classifier, print_performance_metrics_classifier
# from random_forest_c import train_random_forest_classifier, evaluate_model
# from Random_forest_r import train_random_forest_regression, print_performance_metrics_regressor

# # Import the dataset from test.py
# from test import data
# def main():
#     continue_splitting = True
    
#     while continue_splitting:
#         # Perform train-test split
#         user_response = input("Do you want to perform a train-test split on the target column 'sex'? (yes/no): ").strip().lower()
#         if user_response == 'yes':
#             test_size = float(input("Enter the test size (e.g., 0.2 for 20% test data): ").strip())
#             X_train, X_test, y_train, y_test = perform_train_test_split(data, 'sex', test_size)
#             print("Train-test split performed.")
#             print("X_train:\n", X_train.head())  # Print only the head for brevity
#             print("X_test:\n", X_test.head())    # Print only the head for brevity
#             print("y_train:\n", y_train.head())  # Print only the head for brevity
#             print("y_test:\n", y_test.head())    # Print only the head for brevity
#         else:
#             print("Train-test split not performed.")
#             return

#         # Model selection
#         models = ["Decision Tree Classifier", "Linear Regression", "Logistic Regression", "Random Forest Classifier", "Decision Tree Regressor", "Random Forest Regressor"]
#         display_models(models)
#         choice = get_user_choice(len(models))
#         selected_model = models[choice - 1]

#         print(f"You selected: {selected_model}")

#         # Train and evaluate the model
#         if selected_model == "Decision Tree Classifier":
#             model = train_decision_tree_classifier(X_train, y_train)
#             accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
#             print("Decision Tree Classifier Performance:")
#             print("Accuracy:", accuracy)
#             print("Precision:", precision)
#             print("Recall:", recall)
#             print("F1 Score:", f1)
#             print("Confusion Matrix:\n", conf_matrix)
        
#         elif selected_model == "Linear Regression":
#             model = train_linear_regression(X_train, y_train)
#             print("Linear Regression Performance:")
#             print_performance_metrics_regressor(X_test, y_test, 'Linear Regression')
        
#         elif selected_model == "Logistic Regression":
#             model = train_logistic_regression_classifier(X_train, y_train)
#             print("Logistic Regression Performance:")
#             print_performance_metrics_classifier(X_test, y_test, 'Logistic Regression')
        
#         elif selected_model == "Random Forest Classifier":
#             model = train_random_forest_classifier(X_train, y_train)
#             accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
#             print("Random Forest Classifier Performance:")
#             print("Accuracy:", accuracy)
#             print("Precision:", precision)
#             print("Recall:", recall)
#             print("F1 Score:", f1)
#             print("Confusion Matrix:\n", conf_matrix)
        
#         elif selected_model == "Decision Tree Regressor":
#             model = train_decision_tree_regression(X_train, y_train)
#             print("Decision Tree Regressor Performance:")
#             print_performance_metrics_regressor(X_test, y_test, 'Decision Tree Regressor')
        
#         elif selected_model == "Random Forest Regressor":
#             model = train_random_forest_regression(X_train, y_train)
#             print("Random Forest Regressor Performance:")
#             print_performance_metrics_regressor(X_test, y_test, 'Random Forest Regressor')
        
#         # Ask the user if they want to perform the split again
#         repeat_response = input("Do you want to perform another train-test split? (yes/no): ").strip().lower()
#         if repeat_response != 'yes':
#             continue_splitting = False

# if __name__ == "_main_":
#     main()
