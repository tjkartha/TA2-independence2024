
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_decision_tree_classifier(X_train, y_train):
    decision_tree_classifier = DecisionTreeClassifier(random_state=0)
    decision_tree_classifier.fit(X_train, y_train)
    return decision_tree_classifier

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' for multi-class
    recall = recall_score(y_test, y_pred, average='weighted')  # 'weighted' for multi-class
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' for multi-class
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, conf_matrix
