from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class RandomForestRegressorModel:
    def __init__(self, random_state=0):
        self.model = RandomForestRegressor(random_state=random_state)
    
    def fit(self, X_train, y_train):
        """Fit the Random Forest Regressor model to the training data."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions on the test data."""
        return self.model.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        """Calculate and return performance metrics."""
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

# Example usage
# Initialize the model
rf_reg_model = RandomForestRegressorModel()

# Fit the model
rf_reg_model.fit(a_train, s_train)

# Make predictions
y_pred = rf_reg_model.predict(a_test)

# Evaluate the model
mse, r2 = rf_reg_model.evaluate(s_test, y_pred)

# Print the performance metrics
print("Random Forest Regressor Performance:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
