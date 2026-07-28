"""
02_modeling.py — Part B: Predictive modeling, continuing from the same
cleaned data produced by 01_eda.py (reads titanic.csv; never reloads the
raw dataset independently).

Run:
    python 02_modeling.py

Outputs:
    charts/07_decision_tree.png
    charts/08_roc_curves.png
    charts/09_residual_plot.png
    titanic_pipeline.joblib
"""

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from imblearn.over_sampling import SMOTE

CHART_DIR = "charts"


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Continue from the SAME cleaned data (read the committed CSV, no re-load)
# ---------------------------------------------------------------------------
section("0. LOAD CLEANED DATA (from titanic.csv produced by 01_eda.py)")
df = pd.read_csv("titanic.csv")
print("Loaded shape:", df.shape)

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"
NUMERIC_FEATURES = ["age", "sibsp", "parch", "fare", "pclass"]
CATEGORICAL_FEATURES = ["sex", "embarked"]

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------------------------
# 1. STRATIFIED TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
section("1. STRATIFIED TRAIN/TEST SPLIT")
print("Class balance (survived):")
print(y.value_counts(normalize=True).round(3))
print(
    "\nJustification: survival is imbalanced (~59% did not survive vs ~41% did). "
    "A stratified split preserves this ~59/41 ratio in both train and test sets, "
    "so test-set performance isn't skewed by a train/test split that happens to "
    "over- or under-represent survivors, which would make metrics unreliable."
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---------------------------------------------------------------------------
# 2. PREPROCESSING (fit on train only) via ColumnTransformer + Pipeline
# ---------------------------------------------------------------------------
section("2. PREPROCESSING PIPELINE (fit on train only)")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])
print(
    "ColumnTransformer: numeric features -> median-impute + StandardScaler; "
    "categorical features (sex, embarked) -> most-frequent-impute + OneHotEncoder. "
    "Wrapped in a Pipeline per model so every step is fit on X_train only and "
    "applied transform-only to X_test -- never refit on test data."
)

# ---------------------------------------------------------------------------
# 3. TRAIN THREE CLASSIFIERS ON THE SAME SPLIT
# ---------------------------------------------------------------------------
section("3. TRAIN THREE CLASSIFIERS")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

fitted_pipelines = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe
    print(f"Trained: {name}")

# Decision tree visualization
dt_pipe = fitted_pipelines["Decision Tree"]
feature_names = (
    NUMERIC_FEATURES
    + list(dt_pipe.named_steps["preprocessor"]
           .named_transformers_["cat"].named_steps["onehot"]
           .get_feature_names_out(CATEGORICAL_FEATURES))
)
plt.figure(figsize=(20, 10))
plot_tree(
    dt_pipe.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True, max_depth=3, fontsize=8,
)
plt.title("Decision Tree (max_depth=3 shown for readability)")
plt.savefig(f"{CHART_DIR}/07_decision_tree.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved decision tree visualization to charts/07_decision_tree.png")

# ---------------------------------------------------------------------------
# 4. EVALUATE ALL THREE MODELS
# ---------------------------------------------------------------------------
section("4. MODEL EVALUATION (confusion matrix, accuracy, precision, recall, F1, ROC-AUC)")

results = []
plt.figure(figsize=(7, 6))
for name, pipe in fitted_pipelines.items():
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} ---")
    print("Confusion matrix:\n", cm)
    print(f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    results.append({
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc,
    })

plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — all three classifiers")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_roc_curves.png", dpi=110)
plt.close()

comparison_df = pd.DataFrame(results).set_index("model").round(3)
print("\nSide-by-side comparison table:")
print(comparison_df)

# ---------------------------------------------------------------------------
# 5. IMBALANCE HANDLING COMPARISON (using Random Forest)
# ---------------------------------------------------------------------------
section("5. IMBALANCE HANDLING COMPARISON (Random Forest)")
print("Class balance:", y_train.value_counts(normalize=True).round(3).to_dict())

X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

imbalance_results = []

# (a) baseline
rf_base = RandomForestClassifier(n_estimators=200, random_state=42)
rf_base.fit(X_train_pre, y_train)
pred = rf_base.predict(X_test_pre)
imbalance_results.append({
    "strategy": "baseline (no handling)",
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
})

# (b) class_weight='balanced'
rf_bal = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf_bal.fit(X_train_pre, y_train)
pred = rf_bal.predict(X_test_pre)
imbalance_results.append({
    "strategy": "class_weight='balanced'",
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
})

# (c) SMOTE oversampling on the TRAINING FOLD ONLY (no leakage into test)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_pre, y_train)
rf_smote = RandomForestClassifier(n_estimators=200, random_state=42)
rf_smote.fit(X_train_sm, y_train_sm)
pred = rf_smote.predict(X_test_pre)
imbalance_results.append({
    "strategy": "SMOTE (train fold only)",
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
})

imbalance_df = pd.DataFrame(imbalance_results).set_index("strategy").round(3)
print(imbalance_df)
best_strategy = imbalance_df["f1"].idxmax()
print(
    f"\nConclusion: '{best_strategy}' produced the best F1 score among the three "
    "strategies tested. In this dataset the class imbalance is moderate "
    "(~59/41), so the gains from balancing are modest; class_weight='balanced' "
    "typically improves recall on the minority class with no extra data "
    "generation risk, whereas SMOTE's synthetic points can occasionally reduce "
    "precision if they blur the decision boundary. Whichever scores highest "
    "above is the recommended choice for this data."
)

# ---------------------------------------------------------------------------
# 6. HYPERPARAMETER TUNING (GridSearchCV on Random Forest, with OOB score)
# ---------------------------------------------------------------------------
section("6. HYPERPARAMETER TUNING (GridSearchCV + OOB score)")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, None],
    "max_features": ["sqrt", "log2"],
}
rf_for_tuning = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)
grid = GridSearchCV(rf_for_tuning, param_grid, cv=3, scoring="f1", n_jobs=-1)
grid.fit(X_train_pre, y_train)

print("Best params:", grid.best_params_)
print("Best CV F1 score:", round(grid.best_score_, 3))

best_rf = RandomForestClassifier(
    oob_score=True, bootstrap=True, random_state=42, **grid.best_params_
)
best_rf.fit(X_train_pre, y_train)
print("OOB score of refit best model:", round(best_rf.oob_score_, 3))

# ---------------------------------------------------------------------------
# 7. REGRESSION SIDE-TASK: predict fare
# ---------------------------------------------------------------------------
section("7. REGRESSION SIDE-TASK (predict fare)")

reg_features = ["pclass", "age", "sibsp", "parch"]
reg_categorical = ["sex", "embarked"]
X_reg = df[reg_features + reg_categorical]
y_reg = df["fare"]

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), reg_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), reg_categorical),
])

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_pipe = Pipeline(steps=[
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression()),
])
reg_pipe.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_pipe.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
n, p = X_reg_test.shape[0], X_reg_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  Adjusted_R2={adj_r2:.3f}")

residuals = y_reg_test - y_reg_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_pred, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual")
plt.title("Residual plot — fare regression")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/09_residual_plot.png", dpi=110)
plt.close()

pred_median = np.median(y_reg_pred)
residual_spread_low = residuals[y_reg_pred < pred_median].std()
residual_spread_high = residuals[y_reg_pred >= pred_median].std()
hetero_conclusion = (
    "shows heteroscedasticity (residual spread increases noticeably at higher "
    "predicted fares)" if residual_spread_high > 1.5 * residual_spread_low
    else "does not show strong heteroscedasticity (residual spread is roughly "
         "similar across the range of predicted fares)"
)
print(f"\nResidual plot conclusion: the residual pattern {hetero_conclusion}.")

# ---------------------------------------------------------------------------
# 8. FINAL MODEL COMPARISON TABLE + RECOMMENDATION
# ---------------------------------------------------------------------------
section("8. FINAL MODEL COMPARISON TABLE")

print("\nClassification models (accuracy, precision, recall, F1, AUC):")
print(comparison_df)

regression_summary = pd.DataFrame(
    [{"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3), "Adjusted_R2": round(adj_r2, 3)}],
    index=["Linear Regression (fare)"],
)
print("\nRegression model (separate metric group, not on the same scale as classification):")
print(regression_summary)

best_classifier = comparison_df["f1"].idxmax()
best_row = comparison_df.loc[best_classifier]
print(
    f"\nFinal recommendation: deploy the {best_classifier} model. It achieved the "
    f"highest F1 score ({best_row['f1']:.3f}) among the three classifiers, with "
    f"accuracy {best_row['accuracy']:.3f} and AUC {best_row['auc']:.3f}, indicating "
    "the best balance of precision and recall for identifying survivors. Random "
    "Forest models like this one also tend to generalize well on tabular data with "
    "mixed feature types, and the OOB score from tuning above provides an extra "
    "unbiased estimate of generalization without needing a separate validation set."
)

# ---------------------------------------------------------------------------
# 9. SAVE FULL PIPELINE (preprocessing + best estimator) WITH JOBLIB
# ---------------------------------------------------------------------------
section("9. SAVE + RELOAD FULL PIPELINE")

final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        oob_score=True, bootstrap=True, random_state=42, **grid.best_params_
    )),
])
final_pipeline.fit(X_train, y_train)
joblib.dump(final_pipeline, "titanic_pipeline.joblib")
print("Saved complete fitted pipeline to titanic_pipeline.joblib")

reloaded = joblib.load("titanic_pipeline.joblib")
sample_raw = X_test.iloc[:5]
original_preds = final_pipeline.predict(sample_raw)
reloaded_preds = reloaded.predict(sample_raw)
print("Original model predictions on raw sample:", original_preds)
print("Reloaded model predictions on raw sample:", reloaded_preds)
print("Predictions match:", np.array_equal(original_preds, reloaded_preds))

print("\nModeling complete. All charts saved under ./charts/, pipeline saved to titanic_pipeline.joblib.")
