import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix 

import matplotlib.pyplot as plt
import seaborn as sns

# Load the raw Titanic dataset from the official URL
data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Display dataset summary (column data types and missing value counts)
data.info()

# Function to impute missing Age values using median age per Pclass
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
            
    df["Age"] = df.apply(
        lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], 
        axis=1
    )
    return df

# Main preprocessing function: handles missing values, feature encoding, and feature engineering
def preprocess_data(df):
    # Drop identifier columns that do not contribute to predictive power
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    # Impute missing Embarked values and drop the feature
    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Embarked"], inplace=True)

    # Fill missing Age values
    df = fill_missing_ages(df)

    # Encode binary categorical feature (Sex: male=1, female=0)
    df["Sex"] = df["Sex"].map({'male': 1, 'female': 0})

    # Feature Engineering: create family dynamic indicators
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)

    # Impute missing Fare values with median and bin continuous features
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)

    # Fill any residual missing values resulting from binning
    df["FareBin"].fillna(0, inplace=True)
    df["AgeBin"].fillna(0, inplace=True)

    return df

# Apply preprocessing to dataset
data = preprocess_data(data)

# Separate input features (X) and target variable (y)
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Split dataset into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling: normalize all numerical features to range [0, 1] for k-NN
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning function using 5-fold cross-validation
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Find optimal hyperparameters and instantiate the trained model
best_model = tune_model(X_train, y_train)

# Evaluate model performance on unseen test data
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

# Print evaluation results
print(f"Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(matrix)

# Visualize confusion matrix as a heatmap
def plot_model(matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt="d", 
        xticklabels=["Not Survived", "Survived"], 
        yticklabels=["Not Survived", "Survived"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

plot_model(matrix)