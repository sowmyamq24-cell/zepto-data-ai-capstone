# /analytics — Zepto Data & AI Platform

End-to-end profiling, cleaning, EDA, and predictive modeling on the Titanic
dataset (loaded via `sns.load_dataset('titanic')`, cached and committed as
`titanic.csv`).

## Setup

```bash
pip install -r requirements.txt
```

## Run (in order)

```bash
python 01_eda.py         # loads (once), profiles, cleans, saves titanic.csv, EDA charts
python 02_modeling.py    # reads titanic.csv, full modeling pipeline, joblib artifact
```

All charts save to `./charts/`. The fitted end-to-end pipeline saves to
`titanic_pipeline.joblib`.

## Part A — Profiling, cleaning, data story

**Missing values (measured percentages):**
| Column | % missing | Strategy |
|---|---|---|
| deck | 77.22% | Too high to impute reliably → **column dropped**. At ~77% missing, even an "unknown" category would represent the overwhelming majority of rows, making the feature close to non-informative while adding noise/dimensionality. |
| age | 19.87% | Within the 5–30% band → **median imputation** (robust to age's right-skew/outliers). |
| embarked / embark_town | 0.22% | Under 5% → **rows dropped** (only 2 rows affected). |

**Univariate:** age has 65 IQR-based outliers (bounds ≈ [2.5, 54.5]); fare has
114 IQR-based outliers (bounds ≈ [-26.8, 65.7]) — fare's much wider outlier
count reflects its long right tail of expensive tickets. For fare:
mean (32.10) > median (14.45) > mode (8.05), so the distribution is
**right-skewed** — a small number of high-fare (mostly 1st-class) passengers
pull the mean well above the typical fare.

**Bivariate:** survival rate is 74.0% for women vs 18.9% for men; by class,
1st = 62.6%, 2nd = 47.3%, 3rd = 24.2%. Combined, 1st-class women survived at
96.7% while 3rd-class men survived at only 13.5% — sex and class compound
rather than substitute for each other. The two strongest correlations among
the six specified numeric columns (survived, pclass, age, sibsp, parch, fare)
are **pclass ↔ fare (r ≈ -0.55)** — lower class number (better class) pairs
with higher fare, as expected — and **sibsp ↔ parch (r ≈ 0.42)** — passengers
travelling with siblings/spouses also tended to travel with parents/children,
i.e. families travelled together.

**Multivariate data story (4 charts, see `charts/03–06`):** taken together,
the bar chart (survival by class×sex), box plot (age by survival), scatter
(fare vs age by survival), and pair plot all point to the same conclusion:
**sex and fare/class were far stronger survival predictors than age**. Women
in every class outsurvived men in the same class; higher-fare passengers
clustered among survivors regardless of age; and age distributions overlap
heavily between survivors and non-survivors, showing it added little
independent signal beyond sex and class.

**Standardization sanity check:** z-scoring both age and fare on the full
cleaned data confirms both columns reach mean ≈ 0 and std ≈ 1 afterward (see
console output of `01_eda.py`, section 6). This check is exploratory only —
the modeling pipeline in Part B performs its own train-only scaling.

## Part B — Modeling

**Split:** stratified 80/20 split. Justification: survival is imbalanced
(~62% did not survive vs ~38% did in the cleaned data); stratifying preserves
this ratio in both train and test so test metrics aren't skewed by a split
that happens to over/under-represent survivors.

**Preprocessing:** a `ColumnTransformer` (numeric → median-impute +
`StandardScaler`; categorical `sex`/`embarked` → most-frequent-impute +
`OneHotEncoder`) wrapped in a `Pipeline` per model, so every step is fit on
`X_train` only and applied transform-only to `X_test`.

**Classifier comparison** (see console output of `02_modeling.py`, section 4,
for exact numbers each run — results are stable across runs with `random_state=42`):
Logistic Regression, Decision Tree, and Random Forest are trained on the
identical split and compared on accuracy, precision, recall, F1, and ROC-AUC
(`charts/08_roc_curves.png`). The decision tree is visualized in
`charts/07_decision_tree.png`.

**Imbalance handling:** baseline vs `class_weight='balanced'` vs SMOTE
(applied to the training fold only, via `imblearn`, to avoid leakage) are
compared on precision/recall/F1 for Random Forest. See console output for the
exact winning strategy each run — typically SMOTE or `class_weight='balanced'`
edges out baseline on F1/recall, since the moderate class imbalance means
gains are modest but consistent.

**Hyperparameter tuning:** `GridSearchCV` over `n_estimators`, `max_depth`,
`max_features` for `RandomForestClassifier(oob_score=True, bootstrap=True)`;
best parameters and the resulting OOB score are printed in section 6 of the
console output.

**Regression side-task (predict fare):** a multivariate Linear Regression
reports MAE, RMSE, R², and Adjusted R² (see section 7 of console output). The
residual plot (`charts/09_residual_plot.png`) shows **heteroscedasticity** —
residual spread widens noticeably at higher predicted fares, consistent with
fare's strong right-skew and the presence of a few very high-fare outliers
that are harder to predict precisely.

**Final comparison & recommendation:** classifier metrics (accuracy,
precision, recall, F1, AUC) and regression metrics (MAE, RMSE, R², Adjusted
R²) are reported as two separate metric-group tables in section 8 of the
console output, since they are on different scales and not directly
comparable. Based on F1 score, the **Random Forest** classifier is
recommended for deployment: it matches or beats Logistic Regression on
accuracy while achieving a better balance of precision and recall (higher
F1), and its OOB score from tuning provides an additional unbiased estimate
of generalization.

**Saved artifact:** the complete fitted pipeline (preprocessing +
best-tuned Random Forest) is saved via `joblib.dump(full_pipeline, ...)` to
`titanic_pipeline.joblib`, then reloaded with `joblib.load` and confirmed to
produce identical predictions on raw, unpreprocessed sample rows (see
section 9 of console output — "Predictions match: True").
