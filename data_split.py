
import pandas as pd
from sklearn.model_selection import train_test_split

def perform_train_test_split(df, target_column, test_size=0.2):
    X = df.drop(columns=[target_column], axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
