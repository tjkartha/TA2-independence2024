import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def fit(self, X_train, y_train):
        """Fit the Linear Regression model to the training data."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions on the test data."""
        return self.model.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        """Calculate and return the performance metrics."""
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

# Example usage
# Assuming X_train, X_test, y_train, y_test are already defined

# Initialize the model
lin_model = LinearRegressionModel()

# Fit the model
lin_model.fit(X_train, y_train)

# Make predictions
y_pred = lin_model.predict(X_test)

# Evaluate the model
mse, r2 = lin_model.evaluate(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
