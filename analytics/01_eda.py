"""
01_eda.py — Part A: Profiling, Cleaning, and the Data Story
Loads the Titanic dataset ONCE via seaborn, profiles it, cleans it,
saves titanic.csv (the one offline fallback), and produces the full EDA
story (univariate, bivariate, multivariate, standardization check).

Run:
    python 01_eda.py

Outputs:
    titanic.csv                      -- cleaned dataset (offline fallback)
    charts/*.png                     -- all EDA charts
    (all interpretations are printed to stdout AND repeated in README.md)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. LOAD (the one and only load of the raw dataset for the whole module)
# ---------------------------------------------------------------------------
section("1. LOAD + PROFILE")
df = sns.load_dataset("titanic")

print("df.shape:", df.shape)
print("\ndf.info():")
df.info()
print("\ndf.describe():")
print(df.describe())

missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("\nMissing value percentage per affected column:")
print(missing_pct)

# Save the RAW load immediately as the committed offline fallback, per spec.
# (We save the cleaned version at the end under the same filename, since the
# spec's acceptance criterion is that titanic.csv is loadable via
# pd.read_csv and reflects the data the rest of the module works from.)

# ---------------------------------------------------------------------------
# 2. MISSING VALUE HANDLING (threshold rule)
# ---------------------------------------------------------------------------
section("2. MISSING VALUE HANDLING")
print(
    "Rule: <5% missing -> drop rows | 5-30% missing -> impute | "
    "very high (>30%, unreliable to impute) -> drop column or 'missing' category"
)

# age: 19.87% missing -> impute (5-30% band) with median (robust to outliers)
age_pct = missing_pct.get("age", 0.0)
print(f"\n'age' missing = {age_pct}% -> within 5-30% band -> impute with median")
df["age"] = df["age"].fillna(df["age"].median())

# embarked: 0.22% missing -> drop rows (<5% band)
embarked_pct = missing_pct.get("embarked", 0.0)
print(f"'embarked' missing = {embarked_pct}% -> under 5% -> drop those rows")
df = df.dropna(subset=["embarked"])

# embark_town mirrors embarked; same tiny missing rate -> drop rows too (already
# covered by the embarked dropna in practice, but handle explicitly/defensively)
if "embark_town" in df.columns:
    df = df.dropna(subset=["embark_town"])

# deck: ~77% missing -> far too high to impute reliably.
# Decision: DROP the column entirely (not encode-as-missing), because with
# only ~23% of values present, even an "unknown" category would represent the
# overwhelming majority of rows, making the feature close to non-informative
# for modeling while adding noise/dimensionality. Documented here in writing.
deck_pct = missing_pct.get("deck", 0.0)
print(f"'deck' missing = {deck_pct}% -> too high to impute reliably -> DROP the column")
if "deck" in df.columns:
    df = df.drop(columns=["deck"])

print("\nShape after cleaning:", df.shape)
print("Remaining missing values:\n", df.isna().sum()[df.isna().sum() > 0])

# Save the cleaned DataFrame as the one committed offline fallback.
df.to_csv("titanic.csv", index=False)
print("\nSaved cleaned dataset to titanic.csv (offline fallback for the whole module)")

# ---------------------------------------------------------------------------
# 3. UNIVARIATE ANALYSIS: age, fare
# ---------------------------------------------------------------------------
section("3. UNIVARIATE ANALYSIS (age, fare)")


def iqr_outliers(series, name):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    print(f"{name}: Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} bounds=[{lower:.2f}, {upper:.2f}] "
          f"-> {len(outliers)} outliers")
    return len(outliers)


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(df["age"], kde=True, ax=axes[0, 0]).set_title("Age distribution")
sns.boxplot(x=df["age"], ax=axes[0, 1]).set_title("Age boxplot")
sns.histplot(df["fare"], kde=True, ax=axes[1, 0]).set_title("Fare distribution")
sns.boxplot(x=df["fare"], ax=axes[1, 1]).set_title("Fare boxplot")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_univariate_age_fare.png", dpi=110)
plt.close()

age_outliers = iqr_outliers(df["age"], "age")
fare_outliers = iqr_outliers(df["fare"], "fare")

fare_mean, fare_median, fare_mode = df["fare"].mean(), df["fare"].median(), df["fare"].mode()[0]
print(f"\nfare: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
skew_conclusion = (
    "right-skewed (mean > median > mode)" if fare_mean > fare_median > fare_mode
    else "left-skewed (mean < median < mode)" if fare_mean < fare_median < fare_mode
    else "roughly symmetric (mean, median, mode are close)"
)
print(f"Conclusion: fare distribution is {skew_conclusion}.")

# ---------------------------------------------------------------------------
# 4. BIVARIATE ANALYSIS: survival rate by sex, pclass, sex+pclass; correlation
# ---------------------------------------------------------------------------
section("4. BIVARIATE ANALYSIS")

survival_by_sex = df.groupby(df["sex"] == "female")["survived"].mean()
print("Survival rate by sex (boolean mask sex == 'female'):")
print(f"  female: {df[df['sex'] == 'female']['survived'].mean():.3f}")
print(f"  male:   {df[df['sex'] == 'male']['survived'].mean():.3f}")

print("\nSurvival rate by pclass:")
for pc in sorted(df["pclass"].unique()):
    print(f"  pclass {pc}: {df[df['pclass'] == pc]['survived'].mean():.3f}")

print("\nSurvival rate by sex & pclass combined:")
for sex_val in ["female", "male"]:
    for pc in sorted(df["pclass"].unique()):
        mask = (df["sex"] == sex_val) & (df["pclass"] == pc)
        print(f"  {sex_val}, pclass {pc}: {df[mask]['survived'].mean():.3f}")

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df[corr_cols].corr()
print("\nCorrelation matrix (6 specified columns only):")
print(corr_matrix.round(3))

plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap (survived, pclass, age, sibsp, parch, fare)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_correlation_heatmap.png", dpi=110)
plt.close()

# find top-2 absolute off-diagonal correlations
corr_pairs = []
for i, c1 in enumerate(corr_cols):
    for c2 in corr_cols[i + 1:]:
        corr_pairs.append((c1, c2, abs(corr_matrix.loc[c1, c2]), corr_matrix.loc[c1, c2]))
corr_pairs.sort(key=lambda x: x[2], reverse=True)
print("\nTop 2 strongest correlations (by absolute value):")
for c1, c2, abs_val, signed_val in corr_pairs[:2]:
    print(f"  {c1} <-> {c2}: r = {signed_val:.3f}")

# ---------------------------------------------------------------------------
# 5. MULTIVARIATE "DATA STORY" — at least 4 charts, each interpreted
# ---------------------------------------------------------------------------
section("5. MULTIVARIATE DATA STORY (4+ charts)")

# Chart 1: survival rate by class & sex (bar)
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="pclass", y="survived", hue="sex")
plt.title("Survival rate by passenger class and sex")
plt.ylabel("Survival rate")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_survival_by_class_sex.png", dpi=110)
plt.close()
print(
    "Chart 1 (bar): Survival rate by class and sex. Interpretation: women had a "
    "dramatically higher survival rate than men in every class, and 1st-class "
    "women survived at the highest rate of any group, while 3rd-class men had "
    "the lowest. This suggests both 'women and children first' evacuation norms "
    "and class-based access to lifeboats were both at play."
)

# Chart 2: age distribution by survival (box)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="survived", y="age")
plt.title("Age distribution by survival outcome")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_age_by_survival.png", dpi=110)
plt.close()
print(
    "\nChart 2 (box): Age by survival. Interpretation: median age is similar "
    "between survivors and non-survivors, but survivors show a slightly wider "
    "spread toward younger ages, consistent with children being prioritized "
    "during evacuation, though age alone is a weaker signal than sex or class."
)

# Chart 3: fare vs age scatter, colored by survival
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6)
plt.title("Fare vs Age, colored by survival")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_fare_age_survival_scatter.png", dpi=110)
plt.close()
print(
    "\nChart 3 (scatter): Fare vs age, colored by survival. Interpretation: "
    "survivors cluster more densely at higher fare levels, reinforcing that "
    "passengers who paid more (a proxy for class and cabin location) had "
    "better survival odds regardless of age."
)

# Chart 4: pair plot of numeric features colored by survival
pairplot_cols = ["survived", "age", "fare", "pclass"]
pp = sns.pairplot(df[pairplot_cols], hue="survived", diag_kind="hist")
pp.savefig(f"{CHART_DIR}/06_pairplot.png", dpi=110)
plt.close()
print(
    "\nChart 4 (pair plot): Pairwise relationships among survived, age, fare, "
    "pclass. Interpretation: fare and pclass show the clearest visual "
    "separation between survivors and non-survivors of any feature pair here, "
    "while age distributions overlap heavily between the two groups — visually "
    "confirming that socio-economic proxies (fare/class) mattered more than "
    "age for who survived."
)

# ---------------------------------------------------------------------------
# 6. EXPLORATORY STANDARDIZATION CHECK (z-score) — sanity check only
# ---------------------------------------------------------------------------
section("6. EXPLORATORY Z-SCORE STANDARDIZATION CHECK (age, fare)")

for col in ["age", "fare"]:
    before_mean, before_std = df[col].mean(), df[col].std()
    z = (df[col] - before_mean) / before_std
    print(
        f"{col}: BEFORE mean={before_mean:.2f} std={before_std:.2f}  "
        f"-> AFTER (z-score) mean={z.mean():.2f} std={z.std():.2f}"
    )

print(
    "\nNote: this z-score check is purely an EDA-stage sanity check and does "
    "NOT feed into the modeling pipeline in 02_modeling.py, which performs its "
    "own train-only scaling via a scikit-learn Pipeline/ColumnTransformer."
)

print("\nEDA complete. All charts saved under ./charts/. titanic.csv saved for Part B.")
