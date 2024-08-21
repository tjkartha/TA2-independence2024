
import os
from data_split import perform_train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

# Random Forest Classifier
def train_random_forest_classifier(X_train, y_train):
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    return model

# Random Forest Regressor
def train_random_forest_regression(X_train, y_train):
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    return model

# Decision Tree Classifier
def train_decision_tree_classifier(X_train, y_train):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    return model

# Decision Tree Regressor
def train_decision_tree_regression(X_train, y_train):
    model = DecisionTreeRegressor(random_state=123)
    model.fit(X_train, y_train)
    return model

# Linear Regression
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Logistic Regression Classifier
def train_logistic_regression_classifier(X_train, y_train):
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    return model

# Evaluate Classifier Model
def evaluate_model_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    
    return accuracy, precision, recall, f1, conf_matrix

# Evaluate Regressor Model
def print_performance_metrics_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2):", r2)
    
    return mse, rmse, mae, r2

# Display available models
def display_models(models):
    print("Available Models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

