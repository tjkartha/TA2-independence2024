from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import modelbuild

class LogisticRegressionModel:
    def __init__(self, random_state=0):
        self.model = LogisticRegression(random_state=random_state)
    
    def fit(self, X_train, y_train):
        """Fit the Logistic Regression model to the training data."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions on the test data."""
        return self.model.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        """Calculate and return performance metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_true, y_pred)
        return accuracy, precision, recall, f1, conf_matrix

# Example usage
# Initialize the model
log_model = LogisticRegressionModel()

# Fit the model
log_model.fit(modelbuild.X_train,modelbuild.y_train)

# Make predictions
y_pred = log_model.predict(modelbuild.X_test)

# Evaluate the model
accuracy, precision, recall, f1, conf_matrix = log_model.evaluate(modelbuild.y_test, y_pred)

# Print the performance metrics
print("Logistic Regression Classifier Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
