import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, RocCurveDisplay
)

# ============================================================
# 1. GENERATE REALISTIC LOAN DATASET (swap with real CSV later)
# ============================================================
np.random.seed(42)
n = 1000

income = np.random.gamma(shape=5, scale=8000, size=n)
loan_amount = np.random.gamma(shape=4, scale=25000, size=n)
credit_score = np.clip(np.random.normal(650, 80, n), 300, 850)
employment_years = np.random.exponential(5, n).clip(0, 40)
existing_debts = np.random.gamma(shape=2, scale=5000, size=n)
loan_term_months = np.random.choice([12, 24, 36, 60, 120], n)
married = np.random.choice(["Yes", "No"], n)
education = np.random.choice(["Graduate", "Not Graduate"], n, p=[0.7, 0.3])

# Some missing values (realistic - real datasets are messy)
income[np.random.choice(n, 30, replace=False)] = np.nan
credit_score[np.random.choice(n, 20, replace=False)] = np.nan

# Default logic: low credit score, high debt-to-income, short employment -> more risk
debt_to_income = existing_debts / (income + 1)
credit_score_filled = np.nan_to_num(credit_score, nan=650)
debt_to_income_filled = np.nan_to_num(debt_to_income, nan=0.3)

logit = (
    -0.01 * (credit_score_filled - 650)
    + 3 * debt_to_income_filled
    - 0.05 * employment_years
    + 0.00002 * loan_amount
    - 1
)
prob_default = 1 / (1 + np.exp(-logit))
default = (np.random.rand(n) < prob_default).astype(int)   # 1 = defaulted, 0 = paid back

df = pd.DataFrame({
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "existing_debts": existing_debts,
    "loan_term_months": loan_term_months,
    "married": married,
    "education": education,
    "default": default
})

print("Dataset shape:", df.shape)
print("\nClass balance (default rate):\n", df["default"].value_counts(normalize=True))
print("\nMissing values:\n", df.isnull().sum())

# ============================================================
# 2. EDA (quick visual checks)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(x="default", y="credit_score", data=df, ax=axes[0])
axes[0].set_title("Credit Score vs Default")
sns.boxplot(x="default", y="income", data=df, ax=axes[1])
axes[1].set_title("Income vs Default")
sns.countplot(x="default", hue="education", data=df, ax=axes[2])
axes[2].set_title("Education vs Default")
plt.tight_layout()
plt.show()

# ============================================================
# 3. PREPROCESSING PIPELINE (handles missing values + encoding)
# ============================================================
numeric_features = ["income", "loan_amount", "credit_score", "employment_years",
                     "existing_debts", "loan_term_months"]
categorical_features = ["married", "education"]

X = df.drop("default", axis=1)
y = df["default"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ============================================================
# 4. SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 5. COMPARE MODELS (with class_weight='balanced' for imbalance)
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "SVM": SVC(class_weight="balanced", probability=True, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    results[name] = scores.mean()
    print(f"{name} -> CV ROC-AUC: {scores.mean():.3f}")

best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name}")

# ============================================================
# 6. TUNE BEST MODEL (Random Forest example)
# ============================================================
final_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(final_pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

# ============================================================
# 7. FINAL EVALUATION ON TEST SET
# ============================================================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("ROC Curve - Loan Default Prediction")
plt.show()

# ============================================================
# 8. BUSINESS INSIGHT: feature importance
# ============================================================
feature_names = (numeric_features +
    list(best_model.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .named_steps["encoder"]
         .get_feature_names_out(categorical_features)))

importances = pd.Series(
    best_model.named_steps["model"].feature_importances_, index=feature_names
).sort_values(ascending=False)

print("\nTop factors driving default risk:\n", importances.head(8))

plt.figure(figsize=(8, 5))
importances.head(8).plot(kind="barh")
plt.title("Top Factors Driving Loan Default")
plt.gca().invert_yaxis()
plt.show()

# ============================================================
# 9. PREDICT A NEW APPLICANT
# ============================================================
new_applicant = pd.DataFrame({
    "income": [40000], "loan_amount": [150000], "credit_score": [580],
    "employment_years": [1.5], "existing_debts": [20000],
    "loan_term_months": [60], "married": ["No"], "education": ["Not Graduate"]
})
risk_prob = best_model.predict_proba(new_applicant)[0][1]
print(f"\nNew applicant default risk: {risk_prob:.2%}")
