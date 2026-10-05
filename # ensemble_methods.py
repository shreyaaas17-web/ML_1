# ensemble_methods.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 1. BAGGING - train many models in parallel on random subsets
# ============================================================
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=0.8,     # each tree sees 80% of data (randomly sampled)
    random_state=42
)
bagging_model.fit(X_train_scaled, y_train)
print("Bagging (Decision Trees) -> Accuracy:", accuracy_score(y_test, bagging_model.predict(X_test_scaled)))

# Random Forest is essentially Bagging + random feature selection at each split
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train_scaled, y_train)
print("Random Forest (Bagging + feature randomness) -> Accuracy:", accuracy_score(y_test, rf_model.predict(X_test_scaled)))

# ============================================================
# 2. BOOSTING - train sequentially, each model fixes previous errors
# ============================================================
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_model.fit(X_train_scaled, y_train)
print("\nAdaBoost -> Accuracy:", accuracy_score(y_test, adaboost_model.predict(X_test_scaled)))

gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
gb_model.fit(X_train_scaled, y_train)
print("Gradient Boosting -> Accuracy:", accuracy_score(y_test, gb_model.predict(X_test_scaled)))

# ============================================================
# 3. VOTING - simplest ensemble: combine different model types by vote
# ============================================================
voting_model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True))
    ],
    voting='soft'   # 'soft' = average predicted probabilities, 'hard' = majority class vote
)
voting_model.fit(X_train_scaled, y_train)
print("\nVoting Classifier -> Accuracy:", accuracy_score(y_test, voting_model.predict(X_test_scaled)))

# ============================================================
# 4. STACKING - meta-model learns how to combine base models' predictions
# ============================================================
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('nb', GaussianNB()),
    ('svm', SVC(probability=True, random_state=42))
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),   # meta-model
    cv=5
)
stacking_model.fit(X_train_scaled, y_train)
print("Stacking Classifier -> Accuracy:", accuracy_score(y_test, stacking_model.predict(X_test_scaled)))

# ============================================================
# 5. FULL COMPARISON
# ============================================================
results = {
    "Bagging": accuracy_score(y_test, bagging_model.predict(X_test_scaled)),
    "Random Forest": accuracy_score(y_test, rf_model.predict(X_test_scaled)),
    "AdaBoost": accuracy_score(y_test, adaboost_model.predict(X_test_scaled)),
    "Gradient Boosting": accuracy_score(y_test, gb_model.predict(X_test_scaled)),
    "Voting": accuracy_score(y_test, voting_model.predict(X_test_scaled)),
    "Stacking": accuracy_score(y_test, stacking_model.predict(X_test_scaled)),
}

results_df = pd.Series(results).sort_values(ascending=False)
print("\n--- Final Comparison ---")
print(results_df)

plt.figure(figsize=(8, 5))
results_df.plot(kind='barh', color='coral')
plt.xlabel("Test Accuracy")
plt.title("Ensemble Methods Comparison")
plt.gca().invert_yaxis()
plt.xlim(0.8, 1.0)
plt.show()