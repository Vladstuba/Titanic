from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
import pandas as pd

def train_model(X):
    y = X.pop('Survived')
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }
    best_model = None
    best_score = 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        score = accuracy_score(y_valid, y_pred)
        if score > best_score:
            best_score = score
            best_model = model
            name_of_best_model = name
    print(f"Best model: {name_of_best_model} with accuracy: {best_score:.4f}")