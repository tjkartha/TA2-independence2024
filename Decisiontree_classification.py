from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import modelbuild

class DecisionTreeModel:
    def __init__(self, random_state=123):
        self.model = DecisionTreeClassifier(random_state=random_state)
    
    def fit(self, X_train, y_train):
        """Fit the Decision Tree Classifier model to the training data."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Predict the test set using the trained model."""
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
dt_model = DecisionTreeModel()

# Fit the model
dt_model.fit(x_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(x_test)

# Evaluate the model
accuracy_dt, precision_dt, recall_dt, f1_dt, conf_matrix_dt = dt_model.evaluate(y_test, y_pred_dt)

# Print the performance metrics
print("Decision Tree Classifier Performance:")
print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1 Score:", f1_dt)
print("Confusion Matrix:\n", conf_matrix_dt)

