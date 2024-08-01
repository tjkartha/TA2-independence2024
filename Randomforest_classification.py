from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import modelbuild

class RandomForestModel:
    def __init__(self, random_state=0):
        self.model = RandomForestClassifier(random_state=random_state)
    
    def fit(self, X_train, y_train):
        """Fit the Random Forest Classifier model to the training data."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Predict the test set using the trained model."""
        return self.model.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        """Calculate and return performance metrics."""
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, precision, recall, f1, conf_matrix

# Example usage
# Initialize the model
rf_model = RandomForestModel()

# Fit the model
rf_model.fit(modelbuild.X_train, modelbuild.y_train)

# Make predictions
y_pred = rf_model.predict(modelbuild.X_test)

# Evaluate the model
accuracy, precision, recall, f1, conf_matrix = rf_model.evaluate(modelbuild.y_test, modelbuild.y_pred)

# Print the performance metrics
print("Random Forest Classifier Performance:")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

