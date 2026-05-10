import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target) # 0 - malignant, 1 - benign

    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution:")
    print(f"  Benign (1): {sum(y == 1)} ({sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"  Malignant (0): {sum(y == 0)} ({sum(y == 0) / len(y) * 100:.1f}%)")

    return data, X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
    return X_train, X_test, y_train, y_test


def model_evaluation(model, X_test, y_test, model_name, filename):
    y_pred = model.predict(X_test)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant', 'Benign'])
    disp.plot(cmap=plt.cm.Blues)

    folder = "BreastCancer_Visualizations"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename))
    plt.close()
    print(f"Confusion matrix saved: {filename}")

def logistic_regression(X_train, y_train, X_test, y_test):
    lr_model = LogisticRegression(max_iter=10000)
    lr_model.fit(X_train, y_train)
    model_evaluation(lr_model, X_test, y_test, "LOGISTIC REGRESSION", "confusion_matrix_lr.png")

def random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    model_evaluation(rf_model, X_test, y_test, "RANDOM FOREST", "confusion_matrix_rf.png")

def main():
    print("=== WISCONSIN BREAST CANCER - CLASSIFICATION ===")
    data, X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    logistic_regression(X_train, y_train, X_test, y_test)
    random_forest(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()