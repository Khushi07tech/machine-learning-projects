# 🚢 𝓣𝓲𝓽𝓪𝓷𝓲𝓬 𝓢𝓾𝓻𝓿𝓲𝓿𝓪𝓵 𝓟𝓻𝓮𝓭𝓲𝓬𝓽𝓲𝓸𝓷

A machine learning classification pipeline built with Python and `scikit-learn` that predicts passenger survival on the Titanic using feature engineering, missing value imputation, and automated hyperparameter tuning.

---

## 🚀 Key Features

* **Data Preprocessing & Cleaning:** Intelligent imputation of missing demographic and ticket data (e.g., median age calculated per passenger class).
* **Feature Engineering:** Custom feature construction including family group sizes, solitary travel indicators, and quantile-based feature binning.
* **Feature Scaling:** Uniform normalization using `MinMaxScaler` to ensure equal feature weighting for distance calculations.
* **Hyperparameter Optimization:** Automated grid search over multi-parameter spaces using 5-fold cross-validation.
* **Model Evaluation & Visualization:** Detailed performance reporting utilizing precision metrics and an annotated confusion matrix heatmap.

---

## 💻 Core Application Workflows

### 1. Data Cleaning & Missing Value Handling
- Automatically drops non-predictive metadata fields (`PassengerId`, `Name`, `Ticket`, `Cabin`).
- Imputes missing `Embarked` entries using mode calculation (`"S"`).
- Uses a class-aware median strategy (`fill_missing_ages`) to fill missing `Age` values based on `Pclass` categories.

### 2. Feature Transformation & Categorical Encoding
- Maps binary categorical features (`Sex`) to numerical values (`male`: `1`, `female`: `0`).
- Scales numeric parameters using `MinMaxScaler` into a `[0, 1]` bounded range to optimize k-NN distance calculations.

### 3. Custom Feature Engineering
- **`FamilySize`:** Aggregates sibling/spouse count (`SibSp`) and parent/child count (`Parch`).
- **`IsAlone`:** Evaluates whether a passenger traveled without family (`1` if `FamilySize == 0`, else `0`).
- **Feature Binning:** Converts continuous attributes (`Age`, `Fare`) into categorical intervals using `pd.cut` and `pd.qcut`.

### 4. Hyperparameter Tuning with GridSearchCV
- Systematically searches across distance metrics (`euclidean`, `manhattan`, `minkowski`), neighbor counts (`n_neighbors` from 1 to 20), and weighting schemes (`uniform`, `distance`).
- Selects the best-performing `KNeighborsClassifier` estimator based on 5-fold cross-validation performance.

### 5. Model Evaluation & Visualization
- Computes overall classification accuracy on an unseen 25% holdout test set.
- Renders an interactive Seaborn heatmap displaying true positives, true negatives, false positives, and false negatives.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`KNeighborsClassifier`, `GridSearchCV`, `MinMaxScaler`, `train_test_split`)
- **Data Visualization:** Matplotlib, Seaborn

---

## 🔍 Debugging & Data Quality Insights

* **The "100% Accuracy" Anomaly:** During initial testing, the model yielded a 100% evaluation accuracy, indicating artificial target leakage or synthetic labeling rather than true generalization.
* **Root Cause:** The initial dataset was identified as a modified variant of Kaggle's `test.csv`, where target labels had been filled using a deterministic rule (e.g., gender-based survival binary). This created an easily separable boundary for distance-based algorithms.
* **Resolution:** Swapped the data pipeline source to the official ground-truth Titanic training dataset (`train.csv`, 891 records). This eliminated the artificial deterministic pattern and established a realistic evaluation accuracy of **~80%**.

---

## 📁 Repository Structure

```
ml-projects/
├──titanic-survival-prediction/
    ├── main.py
    ├── README.md