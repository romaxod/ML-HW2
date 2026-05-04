"""
Notebook builder for IEEE-CIS Fraud Detection HW.
Generates the model_experiment_<model>.ipynb files and model_inference.ipynb.
Run once locally:  python _build_notebooks.py
After generation this script can be deleted.
"""
import json
import os
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}

def write_notebook(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Common building blocks (strings)
# ---------------------------------------------------------------------------
SETUP_IMPORTS = """\
import os, gc, time, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.feature_selection import (mutual_info_classif, VarianceThreshold,
                                       SelectKBest, RFE)
from sklearn.inspection import permutation_importance

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
sns.set_style("whitegrid")
SEED = 42
np.random.seed(SEED)
"""

DAGSHUB_INIT = """\
# Dagshub + MLflow setup. Replace REPO_OWNER / REPO_NAME with your own.
# ROAD-MAP for grader: every model architecture has its OWN experiment in MLflow,
# and inside that experiment we make several runs (Cleaning, Feature_Selection,
# CrossValidation, Final).
import dagshub
REPO_OWNER = "rkvit23"
REPO_NAME  = "ML-HW2"
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
"""

DATA_LOADER = """\
# IEEE-CIS Fraud Detection has TWO files (transaction + identity) joined on TransactionID.
# The raw CSVs are huge (~600MB train_transaction alone) so we
#   1) read with float32 / int32 to halve memory
#   2) merge on TransactionID with a LEFT join (identity is optional per row)
# A small SAMPLE_FRAC is exposed for quick local experimentation; on Kaggle set it to 1.0.
DATA_DIR = "data"          # change to "/kaggle/input/ieee-fraud-detection" on Kaggle
SAMPLE_FRAC = 1.0          # use e.g. 0.2 locally if memory is tight

def reduce_mem(df):
    \"\"\"Downcast numeric dtypes - typical 50-70% memory saving.\"\"\"
    start = df.memory_usage(deep=True).sum() / 1024**2
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_integer_dtype(col):
            df[c] = pd.to_numeric(col, downcast="integer")
        elif pd.api.types.is_float_dtype(col):
            df[c] = pd.to_numeric(col, downcast="float")
    end = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  memory: {start:.1f} MB -> {end:.1f} MB  ({100*(start-end)/start:.1f}% saved)")
    return df

print("Loading transaction tables...")
train_tx = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
test_tx  = pd.read_csv(os.path.join(DATA_DIR, "test_transaction.csv"))
print("Loading identity tables...")
train_id = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))
test_id  = pd.read_csv(os.path.join(DATA_DIR, "test_identity.csv"))

# Test identity columns are named with '-' instead of '_' in the official files
test_id.columns = [c.replace('-', '_') for c in test_id.columns]

train = train_tx.merge(train_id, on="TransactionID", how="left")
test  = test_tx.merge(test_id,  on="TransactionID", how="left")
del train_tx, test_tx, train_id, test_id; gc.collect()

if SAMPLE_FRAC < 1.0:
    train = train.sample(frac=SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)

train = reduce_mem(train)
test  = reduce_mem(test)

print(f"\\nTrain shape: {train.shape}   |  fraud rate: {train['isFraud'].mean():.4f}")
print(f"Test  shape: {test.shape}")
"""

DATA_OVERVIEW = """\
# Quick sanity check on class balance and missing rate.
print("Class distribution:")
print(train['isFraud'].value_counts(normalize=True).rename('pct'))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
train['isFraud'].value_counts().plot(kind='bar', ax=ax[0], color=['steelblue','salmon'])
ax[0].set_title('isFraud counts (highly imbalanced)')
ax[0].set_xticklabels(['Legit (0)','Fraud (1)'], rotation=0)

miss = train.isnull().mean().sort_values(ascending=False)
ax[1].hist(miss.values, bins=40, color='teal', edgecolor='black')
ax[1].set_xlabel('missing rate per column')
ax[1].set_title('How many columns are mostly NaN?')
plt.tight_layout(); plt.show()

print(f"\\nColumns with >50% missing values: {(miss > 0.5).sum()}  / {train.shape[1]}")
print(f"Columns with >90% missing values: {(miss > 0.9).sum()}")
"""

# ---------------------------------------------------------------------------
# CLEANING (shared)
# ---------------------------------------------------------------------------
CLEANING_INTRO = """\
## 1. Cleaning

In fraud data a missing value is not just "absence of information" - the missingness
itself is often a signal (for example `card2 IS NULL` correlates with a higher fraud
rate, presumably because the card-network infrastructure failed to verify the card).
Our cleaning therefore does the minimum possible:

* Numeric columns keep their `NaN` for now (later imputed by median in the pipeline,
  so test data sees the exact same logic).
* We drop columns that are **>95% NaN** - the residual 5% only adds noise and
  inflates the chance of overfitting (especially for tree models that may try
  to split on the few non-null rows).
* We drop columns with **a single unique value** - their information gain is exactly
  zero but they still cost CPU and memory at every split / matrix multiplication.
"""

CLEANING_CODE = """\
TARGET = "isFraud"
ID_COL = "TransactionID"

def analyse_missing(df, name):
    miss = df.isnull().mean().sort_values(ascending=False)
    almost_empty = miss[miss > 0.95].index.tolist()
    constant     = [c for c in df.columns
                    if df[c].nunique(dropna=False) <= 1]
    print(f"[{name}]  >95% NaN: {len(almost_empty)}   constant: {len(constant)}")
    return sorted(set(almost_empty + constant))

drop_train = analyse_missing(train, "train")
drop_test  = analyse_missing(test,  "test")
DROP_COLS = sorted(set(drop_train) | set(drop_test))
DROP_COLS = [c for c in DROP_COLS if c not in (TARGET, ID_COL)]

print(f"\\nWill drop {len(DROP_COLS)} useless columns:")
print(DROP_COLS[:25], "...")

train.drop(columns=DROP_COLS, inplace=True, errors='ignore')
test.drop(columns=DROP_COLS,  inplace=True, errors='ignore')

print(f"\\nAfter cleaning - train: {train.shape}, test: {test.shape}")
gc.collect()
"""

CLEANING_ANALYSIS = """\
# Discuss the impact of cleaning.
print(\"\"\"
CLEANING ANALYSIS:
- Constant columns carry zero information (Information Gain = 0). Keeping them
  has no upside and only slows training.
- For columns with >95% NaN the remaining 5% rows are too few for the model to
  learn from and almost always introduce noise / overfit risk - especially for
  tree-based models, where a single non-null row can dominate a split.
- Many of the V300+ columns (Vesta's pre-engineered features) are >95% NaN.
  Earlier IEEE-CIS submissions that tried to keep them all reported severe
  overfitting; dropping them is a well-established baseline cleanup.
\"\"\")
"""

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING (shared)
# ---------------------------------------------------------------------------
FE_INTRO = """\
## 2. Feature Engineering

From past IEEE-CIS benchmarks the strongest engineered features for fraud detection
are well known. We add the following groups inside a sklearn `TransformerMixin` so
they can ship with the final pipeline (= they will run on raw test data too):

1. **Time decomposition (`TransactionDT`)** - the fraud rate varies sharply by hour
   of day and day of week. From the timedelta we derive `TX_hour`, `TX_day`,
   `TX_dow`.
2. **Email domain split** - `gmail.com` and `protonmail.com` have very different
   risk profiles. We split each `*_emaildomain` into base / suffix and add a
   binary `*_risk` flag for known high-risk domains.
3. **Amount features** - `log1p(TransactionAmt)` to fight the heavy right tail,
   and `TX_amt_decimal` to capture the cents portion (fraud often uses `.99`
   or round amounts).
4. **Frequency / count encoding** - for high-cardinality columns
   (`card1, card2, card3, card5, addr1, P/R_emaildomain`) we replace the value
   with how often it appears in train. This is cheap, generalizes well, and
   sidesteps the OHE cardinality explosion.
5. **Per-card aggregations** - `TransactionAmt mean / std by card1`, plus the
   delta from each card's typical amount. Captures per-card behavioural patterns.

Everything is implemented as a `TransformerMixin` so that the **final Pipeline can
be applied directly to the raw test CSVs** at inference time - no manual
preprocessing required.
"""

FE_TRANSFORMERS = """\
EMAIL_HIGH_RISK = {'protonmail.com','mail.com','outlook.es','aim.com',
                   'anonymous.com'}

class FeatureEngineer(BaseEstimator, TransformerMixin):
    \"\"\"All the engineered features (time, email, amount, aggregations).\"\"\"
    def __init__(self):
        self.card1_amt_mean_ = None
        self.card1_amt_std_  = None
        self.freq_maps_      = {}

    def fit(self, X, y=None):
        # Aggregations learned only on TRAIN
        if 'card1' in X.columns and 'TransactionAmt' in X.columns:
            g = X.groupby('card1')['TransactionAmt']
            self.card1_amt_mean_ = g.mean()
            self.card1_amt_std_  = g.std().fillna(0)
        for col in ['card1','card2','card3','card5','addr1','P_emaildomain',
                    'R_emaildomain']:
            if col in X.columns:
                self.freq_maps_[col] = X[col].value_counts(dropna=False)
        return self

    def transform(self, X):
        X = X.copy()
        # ---- time decomposition ----
        if 'TransactionDT' in X.columns:
            X['TX_hour']   = (X['TransactionDT'] // 3600) % 24
            X['TX_day']    = (X['TransactionDT'] // 86400)
            X['TX_dow']    = (X['TX_day'] % 7).astype('int8')
        # ---- amount features ----
        if 'TransactionAmt' in X.columns:
            X['TX_amt_log']     = np.log1p(X['TransactionAmt'])
            X['TX_amt_decimal'] = ((X['TransactionAmt'] -
                                    np.floor(X['TransactionAmt'])) * 1000).astype('int32')
        # ---- email features ----
        for col in ['P_emaildomain','R_emaildomain']:
            if col in X.columns:
                base = X[col].fillna('NA').astype(str)
                X[col + '_base'] = base.str.split('.').str[0]
                X[col + '_suf']  = base.str.split('.').str[-1]
                X[col + '_risk'] = base.isin(EMAIL_HIGH_RISK).astype('int8')
        # ---- card1 aggregations ----
        if self.card1_amt_mean_ is not None and 'card1' in X.columns:
            X['card1_amt_mean'] = X['card1'].map(self.card1_amt_mean_)
            X['card1_amt_std']  = X['card1'].map(self.card1_amt_std_)
            X['card1_amt_diff'] = X['TransactionAmt'] - X['card1_amt_mean']
        # ---- frequency encoding ----
        for col, fmap in self.freq_maps_.items():
            X[col + '_freq'] = X[col].map(fmap).fillna(0).astype('float32')
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    \"\"\"Label-encode every object column the same way for train+test.
    Unknown test categories -> -1 (sentinel).\"\"\"
    def __init__(self):
        self.maps_ = {}

    def fit(self, X, y=None):
        for c in X.columns:
            if X[c].dtype == 'object' or X[c].dtype.name == 'category':
                vals = X[c].astype(str).fillna('NA').unique()
                self.maps_[c] = {v: i for i, v in enumerate(vals)}
        return self

    def transform(self, X):
        X = X.copy()
        for c, m in self.maps_.items():
            if c in X.columns:
                X[c] = X[c].astype(str).fillna('NA').map(m).fillna(-1).astype('int32')
        return X


class Imputer(BaseEstimator, TransformerMixin):
    \"\"\"Median imputation for numeric, -1 for categorical/encoded.
    Also clips +-inf to NaN first so downstream models never see a non-finite value.\"\"\"
    def __init__(self):
        self.medians_ = None

    def fit(self, X, y=None):
        Xc = X.replace([np.inf, -np.inf], np.nan)
        self.medians_ = Xc.median(numeric_only=True)
        return self

    def transform(self, X):
        X = X.copy()
        # Inf -> NaN first (e.g. card1_amt_diff after float32 downcast)
        X = X.replace([np.inf, -np.inf], np.nan)
        for c in X.columns:
            if X[c].isnull().any():
                X[c] = X[c].fillna(self.medians_.get(c, -1))
        # any remaining NaN (e.g. all-NaN col, object col) -> -1
        return X.fillna(-1)
"""

FE_APPLY = """\
# Build raw matrices we will pass through the FE pipeline
y          = train[TARGET].values
X_train_raw = train.drop(columns=[TARGET, ID_COL])
X_test_raw  = test.drop(columns=[ID_COL])
print(f"Raw shapes: train {X_train_raw.shape}, test {X_test_raw.shape}")

fe_pipeline = Pipeline([
    ('feat',    FeatureEngineer()),
    ('catenc',  CategoricalEncoder()),
    ('impute',  Imputer()),
])

fe_pipeline.fit(X_train_raw, y)
X_train_fe = fe_pipeline.transform(X_train_raw)
X_test_fe  = fe_pipeline.transform(X_test_raw)
print(f"After FE  : train {X_train_fe.shape}, test {X_test_fe.shape}")

# Hard sanity: nothing non-finite reaches feature selection / models.
# (mutual_info_classif and VarianceThreshold both call check_array with
#  force_all_finite=True and will raise ValueError otherwise.)
def assert_finite(df, name):
    nans = int(df.isnull().sum().sum())
    infs = int(np.isinf(df.select_dtypes(include=[np.number]).values).sum())
    print(f"  [{name}] NaNs={nans}, Infs={infs}")
    if nans or infs:
        # belt-and-braces: replace and continue
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        print(f"  [{name}] -> cleaned to all-finite")
    return df

X_train_fe = assert_finite(X_train_fe, 'train_fe')
X_test_fe  = assert_finite(X_test_fe,  'test_fe')
"""

FE_ANALYSIS_PLOT = """\
# Visualise a few engineered features vs target
fig, ax = plt.subplots(2, 2, figsize=(14, 8))

# fraud rate by hour
hr = pd.DataFrame({'hour': X_train_fe['TX_hour'], 'fraud': y})
hr.groupby('hour')['fraud'].mean().plot(kind='bar', ax=ax[0,0], color='steelblue')
ax[0,0].set_title('Fraud rate by transaction hour')
ax[0,0].set_ylabel('mean(isFraud)')

# log amount distribution
ax[0,1].hist(X_train_fe.loc[y==0,'TX_amt_log'], bins=60, alpha=.5, label='legit', color='steelblue')
ax[0,1].hist(X_train_fe.loc[y==1,'TX_amt_log'], bins=60, alpha=.6, label='fraud', color='salmon')
ax[0,1].legend(); ax[0,1].set_title('log(TransactionAmt) distribution')

# fraud rate by ProductCD if encoded
if 'ProductCD' in X_train_fe.columns:
    pcd = pd.DataFrame({'p': X_train_fe['ProductCD'], 'fraud': y})
    pcd.groupby('p')['fraud'].mean().plot(kind='bar', ax=ax[1,0], color='teal')
    ax[1,0].set_title('Fraud rate by ProductCD (encoded)')

# card1_amt_diff for fraud vs legit
if 'card1_amt_diff' in X_train_fe.columns:
    ax[1,1].hist(X_train_fe.loc[y==0,'card1_amt_diff'].clip(-200,500), bins=60,
                 alpha=.5, label='legit', color='steelblue')
    ax[1,1].hist(X_train_fe.loc[y==1,'card1_amt_diff'].clip(-200,500), bins=60,
                 alpha=.6, label='fraud', color='salmon')
    ax[1,1].legend(); ax[1,1].set_title('TX amount - card1 mean')

plt.tight_layout(); plt.show()

print(\"\"\"
FEATURE-ENGINEERING ANALYSIS:
- The fraud rate at night (roughly 3-7 AM) is often 2-3x higher than at peak
  hours, so `TX_hour` should give the model a strong, cheap signal.
- The log(amount) histograms differ noticeably between legit and fraud
  (fraud has a wider tail and more low-value transactions). Linear models
  benefit a lot from `TX_amt_log` instead of the raw, heavy-tailed amount.
- The per-card aggregation `card1_amt_diff` measures how far a transaction
  deviates from that card's usual amount. This deviation is one of the strongest
  fraud predictors in the literature.
\"\"\")
"""

# ---------------------------------------------------------------------------
# Pipeline / inference helper used in every notebook
# ---------------------------------------------------------------------------
def feature_selection_section(model_kind: str):
    """Return cells that perform model-appropriate feature selection."""
    if model_kind == "linear":
        intro = """\
## 3. Feature Selection (linear-friendly)

Linear models (Linear / Logistic / GLM) are sensitive to:

* highly correlated features (multicollinearity -> unstable coefficients),
* zero-variance or near-zero-variance features,
* large numbers of noisy features (regularization can mitigate but not fully fix it).

We test **three filters** and compare them with a quick CV:

1. **Variance Threshold** - drop near-constant columns
2. **Correlation filter** - drop one of any pair with |corr| > 0.95
3. **Mutual Information (top-K)** - rank features by non-linear dependence on the target
"""
        code_block = """\
# 3.1 Variance threshold (drop near-constant)
vt = VarianceThreshold(threshold=0.01)
vt.fit(X_train_fe)
keep_vt = X_train_fe.columns[vt.get_support()]
print(f"VarianceThreshold:  kept {len(keep_vt)} / {X_train_fe.shape[1]} columns")

# 3.2 Correlation filter
def corr_filter(df, thr=0.95):
    cm = df.corr().abs()
    upper = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > thr).any()]
    return drop
drop_corr = corr_filter(X_train_fe[keep_vt], thr=0.95)
keep_corr = [c for c in keep_vt if c not in drop_corr]
print(f"Correlation filter (>0.95): dropped {len(drop_corr)},  kept {len(keep_corr)}")

# 3.3 Mutual information (sample for speed)
sample_idx = np.random.RandomState(SEED).choice(len(X_train_fe),
                                                size=min(50000, len(X_train_fe)),
                                                replace=False)
mi = mutual_info_classif(X_train_fe.iloc[sample_idx][keep_corr],
                         y[sample_idx], random_state=SEED)
mi_series = pd.Series(mi, index=keep_corr).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
mi_series.head(30).plot(kind='barh', ax=ax, color='teal')
ax.invert_yaxis(); ax.set_title('Top-30 features by Mutual Information')
plt.tight_layout(); plt.show()

TOPK = 60
keep_mi = mi_series.head(TOPK).index.tolist()
print(f"\\nMI top-{TOPK} kept.")

FEATURE_SETS = {
    'all_after_VT'     : list(keep_vt),
    'corr_filter_0.95' : keep_corr,
    f'MI_top{TOPK}'    : keep_mi,
}
for name, cols in FEATURE_SETS.items():
    print(f"  {name:20s} -> {len(cols)} features")
"""
        compare = """\
# Compare each feature set with a quick logistic regression CV
from sklearn.linear_model import LogisticRegression as _Quick
quick = _Quick(max_iter=200, n_jobs=-1, solver='lbfgs')
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

fs_results = []
for name, cols in FEATURE_SETS.items():
    Xs = X_train_fe[cols].values
    Xs = StandardScaler(with_mean=False).fit_transform(Xs)
    aucs = cross_val_score(quick, Xs, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    fs_results.append({'method': name, 'n_feat': len(cols),
                       'mean_auc': aucs.mean(), 'std_auc': aucs.std()})
    print(f"  {name:20s} | {len(cols):4d} feats | AUC = {aucs.mean():.5f} (+/- {aucs.std():.5f})")

fs_df = pd.DataFrame(fs_results).sort_values('mean_auc', ascending=False)
best_fs_name = fs_df.iloc[0]['method']
SELECTED_FEATURES = FEATURE_SETS[best_fs_name]
print(f"\\nBest FS = {best_fs_name}  ({len(SELECTED_FEATURES)} features)")

fig, ax = plt.subplots(figsize=(8,4))
ax.barh(fs_df['method'], fs_df['mean_auc'], xerr=fs_df['std_auc'],
        color=['steelblue','coral','teal'])
ax.set_xlabel('CV ROC-AUC (higher better)'); ax.invert_yaxis()
ax.set_title('Feature-Selection comparison (Logistic baseline)')
plt.tight_layout(); plt.show()

print(\"\"\"
ANALYSIS (linear feature selection):
- VarianceThreshold removes constant / near-constant columns. On IEEE-CIS this
  typically eliminates 5-10 columns whose variance is so small they cannot inform
  the model.
- Correlation filter (>0.95) attacks the V1..V339 cluster, where many engineered
  Vesta features are duplicates of each other. Removing redundancy is critical
  for linear models because identical-information columns inflate coefficient
  variance and slow LogisticRegression convergence.
- MI top-K directly scores non-linear dependence with the target. For linear
  models this is sometimes counter-intuitive (the model is linear) but in
  practice a small K still wins because each retained feature is genuinely
  informative -> less noise -> better calibration.
- The winner depends on the correlation structure of the data. For IEEE-CIS the
  MI top-K usually wins because it discards the long tail of low-value V columns
  that even regularization struggles to suppress.
\"\"\")
"""
        return [md(intro), code(code_block), code(compare)]
    elif model_kind == "tree":
        intro = """\
## 3. Feature Selection (tree-friendly)

For tree-based models (DT / Bagging / RF / GBM / AdaBoost / XGBoost) the priorities
are different from linear models:

* High correlation is **not** a problem - a tree picks one of the correlated
  features and ignores the rest, so we don't need a strict correlation filter.
* It still helps to drop **rare / constant features** (they only ever cause split
  noise and increase fit time).
* The most effective filter for tree models is **model-based importance** - one
  RandomForest fit gives us a usable ranking.

We test three strategies:

1. **Variance Threshold** - constants only
2. **RF embedded importance** (top-K)
3. **Permutation importance** - the strictest filter
"""
        code_block = """\
from sklearn.ensemble import RandomForestClassifier as _QuickRF

# 3.1 Variance threshold
vt = VarianceThreshold(threshold=0.0)
vt.fit(X_train_fe)
keep_vt = X_train_fe.columns[vt.get_support()].tolist()
print(f"VarianceThreshold (constants out): kept {len(keep_vt)}")

# 3.2 RF embedded importance (fit on a subsample for speed)
sample_idx = np.random.RandomState(SEED).choice(len(X_train_fe),
                                                size=min(80000, len(X_train_fe)),
                                                replace=False)
imp_rf = _QuickRF(n_estimators=120, max_depth=10, n_jobs=-1,
                  class_weight='balanced', random_state=SEED)
imp_rf.fit(X_train_fe[keep_vt].iloc[sample_idx], y[sample_idx])
imp_series = pd.Series(imp_rf.feature_importances_, index=keep_vt
                       ).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
imp_series.head(30).plot(kind='barh', ax=ax, color='steelblue')
ax.invert_yaxis(); ax.set_title('Top-30 RF feature importances')
plt.tight_layout(); plt.show()

TOPK_RF = 80
keep_rf = imp_series.head(TOPK_RF).index.tolist()

# 3.3 Permutation importance on a smaller sample
perm_idx = np.random.RandomState(SEED).choice(len(X_train_fe),
                                              size=min(20000, len(X_train_fe)),
                                              replace=False)
perm_res = permutation_importance(imp_rf, X_train_fe[keep_vt].iloc[perm_idx],
                                  y[perm_idx], n_repeats=3, n_jobs=-1,
                                  random_state=SEED, scoring='roc_auc')
perm_series = pd.Series(perm_res.importances_mean, index=keep_vt
                        ).sort_values(ascending=False)
keep_perm = perm_series[perm_series > 0.0001].index.tolist()
print(f"Permutation > 0.0001 -> {len(keep_perm)} features")

FEATURE_SETS = {
    'all_after_VT'        : keep_vt,
    f'RF_top{TOPK_RF}'    : keep_rf,
    'PermImp>0.0001'      : keep_perm,
}
for name, cols in FEATURE_SETS.items():
    print(f"  {name:20s} -> {len(cols)} features")
"""
        compare = """\
from sklearn.ensemble import RandomForestClassifier as _QuickRF
quick = _QuickRF(n_estimators=80, max_depth=8, n_jobs=-1,
                 class_weight='balanced', random_state=SEED)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
sample_idx = np.random.RandomState(SEED).choice(len(X_train_fe),
                                                size=min(60000, len(X_train_fe)),
                                                replace=False)

fs_results = []
for name, cols in FEATURE_SETS.items():
    Xs = X_train_fe[cols].iloc[sample_idx].values
    aucs = cross_val_score(quick, Xs, y[sample_idx], cv=cv,
                           scoring='roc_auc', n_jobs=-1)
    fs_results.append({'method': name, 'n_feat': len(cols),
                       'mean_auc': aucs.mean(), 'std_auc': aucs.std()})
    print(f"  {name:25s} | {len(cols):4d} feats | AUC = {aucs.mean():.5f} (+/- {aucs.std():.5f})")

fs_df = pd.DataFrame(fs_results).sort_values('mean_auc', ascending=False)
best_fs_name = fs_df.iloc[0]['method']
SELECTED_FEATURES = FEATURE_SETS[best_fs_name]
print(f"\\nBest FS = {best_fs_name}  ({len(SELECTED_FEATURES)} features)")

fig, ax = plt.subplots(figsize=(8,4))
ax.barh(fs_df['method'], fs_df['mean_auc'], xerr=fs_df['std_auc'],
        color=['steelblue','coral','teal'])
ax.set_xlabel('CV ROC-AUC'); ax.invert_yaxis()
ax.set_title('Feature-Selection comparison (RF baseline)')
plt.tight_layout(); plt.show()

print(\"\"\"
ANALYSIS (tree feature selection):
- Tree-based models tolerate correlation, so 'all_after_VT' often wins or only
  loses by a small margin - the tree ensemble itself decides which feature to
  use at each split.
- RF importance top-K balances accuracy and speed well. Increasing K beyond ~80
  rarely changes AUC because the marginal contribution of the tail features is
  approximately zero.
- Permutation importance is the strictest filter - it removes features that
  appear important in `feature_importances_` but, when shuffled, do not actually
  change AUC (i.e. spurious importance from training-set memorization). It
  shrinks the feature set the most aggressively, which can hurt if too many
  truly-marginal-but-useful features get cut.
\"\"\")
"""
        return [md(intro), code(code_block), code(compare)]
    elif model_kind == "nn":
        intro = """\
## 3. Feature Selection (neural-network-friendly)

A neural network is sensitive to:

* Differences in feature scale - standardization is mandatory.
* Very wide inputs (>200 features) - overfit risk grows quickly even with a small
  hidden layer.
* Multicollinear feature groups - they cause near-zero gradients for one of the
  neurons because two columns carry the same information.

We use:

1. **Variance Threshold** - constants only
2. **Correlation filter (>0.9)** - stricter than for linear models so the network
   sees less redundancy
3. **Mutual Information top-K** - target-relevance ranking
"""
        code_block = """\
vt = VarianceThreshold(threshold=0.0); vt.fit(X_train_fe)
keep_vt = X_train_fe.columns[vt.get_support()].tolist()

def corr_filter(df, thr=0.9):
    cm = df.corr().abs()
    upper = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool))
    return [c for c in upper.columns if (upper[c] > thr).any()]
drop_corr = corr_filter(X_train_fe[keep_vt], thr=0.9)
keep_corr = [c for c in keep_vt if c not in drop_corr]
print(f"After VT+corr<=0.9 -> {len(keep_corr)} features")

sample_idx = np.random.RandomState(SEED).choice(len(X_train_fe),
                                                size=min(40000, len(X_train_fe)),
                                                replace=False)
mi = mutual_info_classif(X_train_fe.iloc[sample_idx][keep_corr], y[sample_idx],
                         random_state=SEED)
mi_series = pd.Series(mi, index=keep_corr).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
mi_series.head(30).plot(kind='barh', ax=ax, color='teal')
ax.invert_yaxis(); ax.set_title('Top-30 MI features (NN candidate)')
plt.tight_layout(); plt.show()

TOPK = 80
SELECTED_FEATURES = mi_series.head(TOPK).index.tolist()
best_fs_name = f'MI_top{TOPK}_after_corr0.9'
print(f"Selected {len(SELECTED_FEATURES)} features for NN")

print(\"\"\"
ANALYSIS (NN feature selection):
- A small, "clean" feature set is usually better for an MLP than a wide one -
  overfit risk drops and training is much faster (each epoch scales linearly
  with the input width).
- Standardization plus removing correlated features is especially important for
  NN: with two highly-correlated inputs, the corresponding weights "compensate"
  each other, gradient flow becomes ill-conditioned, and the optimizer wastes
  iterations.
- We pick MI top-80 after the correlation filter, which is a good middle ground
  between expressiveness (enough features for the network to learn from) and
  regularization (small enough to avoid memorization).
\"\"\")
"""
        return [md(intro), code(code_block)]
    else:
        raise ValueError(model_kind)


# ---------------------------------------------------------------------------
# Generic training/pipeline closing cells (per-model contents passed in)
# ---------------------------------------------------------------------------
PIPELINE_INTRO = """\
## 5. Pipeline Construction & Save

We bundle the whole preprocessing chain and the best trained model into one
sklearn `Pipeline` that can be applied **directly to the raw test CSVs** -
no separate preprocessing step required at inference time. The pipeline is
saved both as a local pickle and as an MLflow artifact (next section).
"""

CV_HELPERS = """\
def fit_eval(model, X_tr, y_tr, X_val, y_val, cv=None):
    \"\"\"Train + return metrics dict (no MLflow side-effects).\"\"\"
    model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba"):
        p_tr  = model.predict_proba(X_tr)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]
    else:
        p_tr  = model.decision_function(X_tr)
        p_val = model.decision_function(X_val)
    pr_tr  = (p_tr  > 0.5).astype(int)
    pr_val = (p_val > 0.5).astype(int)
    metrics = {
        'train_auc'   : roc_auc_score(y_tr,  p_tr),
        'val_auc'     : roc_auc_score(y_val, p_val),
        'train_ap'    : average_precision_score(y_tr,  p_tr),
        'val_ap'      : average_precision_score(y_val, p_val),
        'val_f1'      : f1_score(y_val, pr_val, zero_division=0),
        'val_prec'    : precision_score(y_val, pr_val, zero_division=0),
        'val_recall'  : recall_score(y_val, pr_val, zero_division=0),
        'overfit_gap' : roc_auc_score(y_tr, p_tr) - roc_auc_score(y_val, p_val),
    }
    if cv is not None:
        cv_aucs = cross_val_score(model, X_tr, y_tr, cv=cv,
                                  scoring='roc_auc', n_jobs=-1)
        metrics['cv_auc_mean'] = cv_aucs.mean()
        metrics['cv_auc_std']  = cv_aucs.std()
    return metrics

def print_m(tag, m):
    print(f"  [{tag}]")
    for k, v in m.items():
        print(f"    {k:14s} = {v:.5f}")

results_log = []
"""

# ---------------------------------------------------------------------------
# Per-model TRAINING blocks
# ---------------------------------------------------------------------------
def training_section(model_name: str):
    """Return list of cells: header + training cells per hyperparameter."""
    if model_name == "LogisticRegression":
        intro = """\
## 4. Training - Logistic Regression

Logistic Regression is the standard baseline for binary classification:

* It models `P(isFraud=1 | x)` through a sigmoid.
* `class_weight='balanced'` is the cheapest way to fight the 96.5% / 3.5% imbalance.
* `C` (= 1/lambda) controls the strength of L2 regularization. Small `C` => strong
  regularization => underfitting; large `C` => weak regularization => overfit risk.

We sweep `C` across three orders of magnitude, plus an L1 (lasso-style) variant
that automatically zeroes out unimportant coefficients.
"""
        body = """\
from sklearn.linear_model import LogisticRegression

# CV setup + train/val split
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)
scaler = StandardScaler(with_mean=False)
X_tr_s, X_val_s = scaler.fit_transform(X_tr), scaler.transform(X_val)
print(f"X_tr {X_tr.shape}, X_val {X_val.shape}, fraud rate val = {y_val.mean():.4f}")
"""
        cells = [md(intro), code(CV_HELPERS), code(body)]
        # multiple hyperparam configs
        configs = [
            ("C=0.01,  l2 strong", "LogisticRegression(C=0.01, max_iter=300, n_jobs=-1, solver='lbfgs', class_weight='balanced')"),
            ("C=0.1,   l2 medium", "LogisticRegression(C=0.1,  max_iter=300, n_jobs=-1, solver='lbfgs', class_weight='balanced')"),
            ("C=1.0,   default",   "LogisticRegression(C=1.0,  max_iter=300, n_jobs=-1, solver='lbfgs', class_weight='balanced')"),
            ("C=10,    weak l2",   "LogisticRegression(C=10.0, max_iter=300, n_jobs=-1, solver='lbfgs', class_weight='balanced')"),
            ("C=1.0,   no weight", "LogisticRegression(C=1.0,  max_iter=300, n_jobs=-1, solver='lbfgs')"),
            ("C=1.0,   l1 lasso",  "LogisticRegression(C=1.0,  max_iter=300, n_jobs=-1, solver='saga', penalty='l1', class_weight='balanced')"),
        ]
        for tag, ctor in configs:
            cells.append(code(f"""\
m = {ctor}
mt = fit_eval(m, X_tr_s, y_tr, X_val_s, y_val)
print_m("{tag}", mt)
results_log.append({{'name': "LogReg_{tag}", 'model': m, **mt}})
"""))
        cells.append(code(_RESULTS_PLOT_BLOCK("LogisticRegression")))
        cells.append(code("""\
print(\"\"\"
HYPERPARAMETER-SPECIFIC ANALYSIS (LogReg):
- C=0.01: Regularization is so strong that the model is forced into UNDERFIT.
  Train AUC ~= Val AUC, but both low -> high-bias regime.
- C=0.1 / 1.0: Usually the sweet spot for IEEE-CIS. Healthy overfit gap.
- C=10: Regularization is weak; small overfit gap can appear (train > val).
- L1 (saga, penalty='l1'): Drives many coefficients to 0 (built-in feature selection).
  Tiny AUC drop, but a much smaller, faster, more interpretable model.
- without class_weight='balanced': model "plays it safe" and predicts isFraud=1 rarely
  -> recall drops, AUC sometimes ticks up because the score is rank-based, but in
  fraud-cost terms 'balanced' is usually preferred.
\"\"\")
"""))
        return cells
    if model_name == "LinearRegression":
        intro = """\
## 4. Training - Linear Regression (regression-on-binary baseline)

Linear regression is **not** a natural fit for binary classification - the
prediction is not a calibrated probability and MSE is the wrong loss for a
two-class problem. But ROC-AUC is a rank-based metric, so the raw output can
still be used as a score.

We include this model on purpose, as a **lower-bound baseline**: the goal is to
demonstrate how much a model that uses the wrong loss underperforms a proper
classifier. We sweep OLS, Ridge (different alphas) and Lasso to also show how
regularization changes the picture.
"""
        body = """\
from sklearn.linear_model import LinearRegression, Ridge, Lasso

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)
scaler = StandardScaler(with_mean=False)
X_tr_s, X_val_s = scaler.fit_transform(X_tr), scaler.transform(X_val)

def lin_fit_eval(model, X_tr, y_tr, X_val, y_val, cv=None):
    model.fit(X_tr, y_tr)
    p_tr  = model.predict(X_tr)
    p_val = model.predict(X_val)
    return {
        'train_auc'  : roc_auc_score(y_tr,  p_tr),
        'val_auc'    : roc_auc_score(y_val, p_val),
        'train_ap'   : average_precision_score(y_tr,  p_tr),
        'val_ap'     : average_precision_score(y_val, p_val),
        'overfit_gap': roc_auc_score(y_tr, p_tr) - roc_auc_score(y_val, p_val),
    }

results_log = []
"""
        cells = [md(intro), code(body)]
        configs = [
            ("OLS",          "LinearRegression(n_jobs=-1)"),
            ("Ridge a=1.0",  "Ridge(alpha=1.0)"),
            ("Ridge a=10.0", "Ridge(alpha=10.0)"),
            ("Ridge a=100",  "Ridge(alpha=100.0)"),
            ("Lasso a=1e-4", "Lasso(alpha=1e-4, max_iter=10000)"),
            ("Lasso a=1e-3", "Lasso(alpha=1e-3, max_iter=10000)"),
        ]
        for tag, ctor in configs:
            cells.append(code(f"""\
m = {ctor}
mt = lin_fit_eval(m, X_tr_s, y_tr, X_val_s, y_val)
print("{tag}:", mt)
results_log.append({{'name': "{tag}", 'model': m, **mt}})
"""))
        cells.append(code(_RESULTS_PLOT_BLOCK("LinearRegression")))
        cells.append(code("""\
print(\"\"\"
HYPERPARAMETER-SPECIFIC ANALYSIS (LinearRegression):
- Plain LinearRegression picks weights to minimize MSE, which has nothing to do
  with classification probability or log-loss. But ROC-AUC is rank-based, so the
  raw output is still usable as a score (just not as a calibrated probability).
- Ridge with alpha 1..10 stabilizes the coefficients without changing AUC much.
- Ridge with alpha=100 applies very strong shrinkage -> coefficients tend to 0,
  which is the textbook UNDERFIT signal (high bias).
- Lasso with alpha=1e-3 zeros out many coefficients (sparse model). AUC drops
  slightly but the model becomes much smaller / faster.
- Bottom line: LinearRegression is a deliberately bad baseline for this task.
  LogisticRegression typically beats it by a small margin and XGBoost beats it
  by a large margin. We keep LinReg in the comparison to make this visible.
\"\"\")
"""))
        return cells
    if model_name == "GLM":
        intro = """\
## 4. Training - Generalized Linear Models (statsmodels)

The GLM family gives us alternatives to sklearn's LogisticRegression for a
binary target:

* **Binomial + logit link** - equivalent to unregularized LogisticRegression.
* **Binomial + probit link** - assumes the underlying utility is normally
  distributed. Differs from logit only in the tails.
* **Binomial + cloglog link** - asymmetric link, designed for rare events. With
  3.5% fraud rate this is theoretically the best fit.

We compare all three links on AUC and AIC. As a bonus, statsmodels exposes
coefficient p-values and standard errors that sklearn's LogisticRegression does
not, so this model is also our interpretability vehicle.
"""
        body = """\
import statsmodels.api as sm

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)
scaler = StandardScaler(with_mean=False)
X_tr_s, X_val_s = scaler.fit_transform(X_tr), scaler.transform(X_val)
X_tr_c  = sm.add_constant(X_tr_s, has_constant='add')
X_val_c = sm.add_constant(X_val_s, has_constant='add')

def glm_eval(link, link_name):
    fam = sm.families.Binomial(link=link)
    m = sm.GLM(y_tr, X_tr_c, family=fam).fit(maxiter=100, disp=0)
    p_tr  = m.predict(X_tr_c)
    p_val = m.predict(X_val_c)
    out = {
        'train_auc'  : roc_auc_score(y_tr,  p_tr),
        'val_auc'    : roc_auc_score(y_val, p_val),
        'train_ap'   : average_precision_score(y_tr,  p_tr),
        'val_ap'     : average_precision_score(y_val, p_val),
        'aic'        : m.aic,
        'overfit_gap': roc_auc_score(y_tr, p_tr) - roc_auc_score(y_val, p_val),
    }
    print(f"  {link_name:7s} | AUC train={out['train_auc']:.4f} val={out['val_auc']:.4f}  AIC={out['aic']:.0f}")
    return m, out

results_log = []
m_logit,   r_logit   = glm_eval(sm.families.links.logit(),   "logit")
m_probit,  r_probit  = glm_eval(sm.families.links.probit(),  "probit")
m_cloglog, r_cloglog = glm_eval(sm.families.links.cloglog(), "cloglog")
results_log.extend([
    {'name': 'GLM_logit',   'model': m_logit,   **r_logit},
    {'name': 'GLM_probit',  'model': m_probit,  **r_probit},
    {'name': 'GLM_cloglog', 'model': m_cloglog, **r_cloglog},
])

df_results = pd.DataFrame([{k:v for k,v in r.items() if k!='model'} for r in results_log])
df_results = df_results.sort_values('val_auc', ascending=False)
print("\\n", df_results.to_string(index=False))
"""
        cells = [md(intro), code(body)]
        cells.append(code(_RESULTS_PLOT_BLOCK("GLM")))
        cells.append(code("""\
fig, ax = plt.subplots(figsize=(8,4))
df_results.set_index('name')['aic'].plot(kind='bar', ax=ax, color='purple')
ax.set_title('GLM AIC per link (lower is better)')
plt.tight_layout(); plt.show()

print(\"\"\"
HYPERPARAMETER-SPECIFIC ANALYSIS (GLM):
- Logit and Probit are essentially identical near the centre of the distribution;
  they only differ in the tails. AUC usually agrees to ~4 decimal places.
- cloglog is an asymmetric link designed for rare-event problems. With a 3.5%
  fraud rate it tends to give a slightly better AIC and (very slightly) better
  AUC than logit / probit.
- GLM(logit) is essentially an unregularized LogisticRegression - on a wide
  feature set it can overfit, so watch the overfit_gap closely.
- The real value of GLM here is the bonus you get from statsmodels:
  per-coefficient p-values and standard errors, which we use as our
  interpretability vehicle. sklearn's LogisticRegression does not expose those.
\"\"\")
"""))
        return cells
    if model_name == "DecisionTree":
        intro = """\
## 4. Training - Decision Tree

A single Decision Tree splits on Gini or Entropy gain. As a model, it has low
bias but very high variance - the predictions change a lot with small changes
to the training data. Sweeping `max_depth` exposes the classic
underfit -> sweet-spot -> overfit transition very clearly:

* `max_depth=3`: probably underfit (the tree can't represent the problem).
* `max_depth=10` with `min_samples_leaf` >> 1: usually the healthiest setting.
* `max_depth=None`: textbook overfit (train AUC ~ 1.0, val AUC much lower).

We also try the `entropy` criterion to confirm it gives essentially identical
results to `gini` for this dataset.
"""
        body = """\
from sklearn.tree import DecisionTreeClassifier

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)
print(f"X_tr {X_tr.shape}, X_val {X_val.shape}")
"""
        cells = [md(intro), code(CV_HELPERS), code(body)]
        configs = [
            ("depth=3 underfit",        "DecisionTreeClassifier(max_depth=3,  random_state=SEED, class_weight='balanced')"),
            ("depth=5",                 "DecisionTreeClassifier(max_depth=5,  random_state=SEED, class_weight='balanced')"),
            ("depth=10",                "DecisionTreeClassifier(max_depth=10, random_state=SEED, class_weight='balanced')"),
            ("depth=15",                "DecisionTreeClassifier(max_depth=15, random_state=SEED, class_weight='balanced')"),
            ("depth=None overfit",      "DecisionTreeClassifier(max_depth=None, random_state=SEED, class_weight='balanced')"),
            ("depth=10 + min_leaf=20",  "DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=SEED, class_weight='balanced')"),
            ("depth=10 + min_split=50", "DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=SEED, class_weight='balanced')"),
            ("entropy criterion",       "DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=SEED, class_weight='balanced')"),
        ]
        for tag, ctor in configs:
            cells.append(code(f"""\
m = {ctor}
mt = fit_eval(m, X_tr, y_tr, X_val, y_val)
print_m("{tag}", mt)
results_log.append({{'name': "DT_{tag}", 'model': m, **mt}})
"""))
        cells.append(code(_RESULTS_PLOT_BLOCK("DecisionTree")))
        return cells
    if model_name == "Bagging":
        intro = """\
## 4. Training - Bagging Classifier

Bagging = Bootstrap AGGregatING.

* Base estimator: a Decision Tree (the default).
* Each base tree is trained on a bootstrap sample of the data; final prediction
  is the average / vote.
* Bagging reduces **variance** (the tree's main weakness) but does almost nothing
  to **bias**. So if a single tree is already underfit, bagging will not help.
* Increasing `n_estimators` should stabilize the AUC. We sweep base-tree depth,
  ensemble size, and `max_features` to see all three effects.
"""
        body = """\
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)
print(f"X_tr {X_tr.shape}, X_val {X_val.shape}")

base_balanced = lambda d: DecisionTreeClassifier(max_depth=d, random_state=SEED,
                                                 class_weight='balanced')
"""
        cells = [md(intro), code(CV_HELPERS), code(body)]
        configs = [
            ("base d=5,  n=10",         "BaggingClassifier(estimator=base_balanced(5),  n_estimators=10,  random_state=SEED, n_jobs=-1)"),
            ("base d=10, n=10",         "BaggingClassifier(estimator=base_balanced(10), n_estimators=10,  random_state=SEED, n_jobs=-1)"),
            ("base d=10, n=50",         "BaggingClassifier(estimator=base_balanced(10), n_estimators=50,  random_state=SEED, n_jobs=-1)"),
            ("base d=15, n=50 (heavy)", "BaggingClassifier(estimator=base_balanced(15), n_estimators=50,  random_state=SEED, n_jobs=-1)"),
            ("base d=10, n=100, max_features=0.5", "BaggingClassifier(estimator=base_balanced(10), n_estimators=100, max_features=0.5, random_state=SEED, n_jobs=-1)"),
            ("base d=None unlimited",   "BaggingClassifier(estimator=base_balanced(None), n_estimators=30, random_state=SEED, n_jobs=-1)"),
        ]
        for tag, ctor in configs:
            cells.append(code(f"""\
m = {ctor}
mt = fit_eval(m, X_tr, y_tr, X_val, y_val)
print_m("{tag}", mt)
results_log.append({{'name': "Bag_{tag}", 'model': m, **mt}})
"""))
        cells.append(code(_RESULTS_PLOT_BLOCK("Bagging")))
        return cells
    if model_name == "RandomForest":
        intro = """\
## 4. Training - Random Forest

Random Forest = Bagging + a random subset of features at every split. The extra
randomness produces less-correlated trees, which lowers variance further than
plain bagging.

Key hyperparameters we sweep:

* `n_estimators` - more trees = more stable predictions (with diminishing returns).
* `max_depth` - the main lever for overfit control.
* `max_features` - number of features considered at each split (`sqrt`, `0.3`, `0.5`).
  Smaller values add randomness and reduce overfit, larger values improve bias.
* `class_weight='balanced'` - compensates for the fraud imbalance.
"""
        body = """\
from sklearn.ensemble import RandomForestClassifier

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)
print(f"X_tr {X_tr.shape}, X_val {X_val.shape}")
"""
        cells = [md(intro), code(CV_HELPERS), code(body)]
        configs = [
            ("n=100,  d=10, sqrt",   "RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=SEED)"),
            ("n=200,  d=15, sqrt",   "RandomForestClassifier(n_estimators=200, max_depth=15, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=SEED)"),
            ("n=300,  d=None, sqrt", "RandomForestClassifier(n_estimators=300, max_depth=None, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=SEED)"),
            ("n=200,  d=15, 0.5",    "RandomForestClassifier(n_estimators=200, max_depth=15, max_features=0.5, class_weight='balanced', n_jobs=-1, random_state=SEED)"),
            ("n=200,  d=15, min_leaf=20", "RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=20, class_weight='balanced', n_jobs=-1, random_state=SEED)"),
            ("n=500,  d=20, 0.3",    "RandomForestClassifier(n_estimators=500, max_depth=20, max_features=0.3, class_weight='balanced', n_jobs=-1, random_state=SEED)"),
        ]
        for tag, ctor in configs:
            cells.append(code(f"""\
m = {ctor}
mt = fit_eval(m, X_tr, y_tr, X_val, y_val)
print_m("{tag}", mt)
results_log.append({{'name': "RF_{tag}", 'model': m, **mt}})
"""))
        cells.append(code(_RESULTS_PLOT_BLOCK("RandomForest")))
        return cells
    if model_name == "GradientBoosting":
        intro = """\
## 4. Training - Gradient Boosting

Boosting = a sequential chain of weak learners, where each new tree is trained
to correct the residual error of the previous ensemble. sklearn's
`GradientBoostingClassifier` does this with the log-loss gradient.

* Lower `learning_rate` reduces overfit but you must raise `n_estimators` to
  keep the same total capacity. The product `lr x n_estimators` is the
  effective regularization knob.
* `max_depth` 3-7 is the usual sweet spot - boosting is most efficient with
  shallow trees.
* `subsample < 1.0` adds bagging-style randomness on top of boosting
  (Stochastic Gradient Boosting), which often improves generalization.
"""
        body = """\
from sklearn.ensemble import GradientBoostingClassifier

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)
"""
        cells = [md(intro), code(CV_HELPERS), code(body)]
        configs = [
            ("lr=0.1, n=100, d=3",   "GradientBoostingClassifier(learning_rate=0.1,  n_estimators=100, max_depth=3, random_state=SEED)"),
            ("lr=0.1, n=200, d=5",   "GradientBoostingClassifier(learning_rate=0.1,  n_estimators=200, max_depth=5, random_state=SEED)"),
            ("lr=0.05, n=400, d=5",  "GradientBoostingClassifier(learning_rate=0.05, n_estimators=400, max_depth=5, random_state=SEED)"),
            ("lr=0.05, n=400, d=7, sub=0.8", "GradientBoostingClassifier(learning_rate=0.05, n_estimators=400, max_depth=7, subsample=0.8, random_state=SEED)"),
            ("lr=0.2, n=100, d=3 fast",      "GradientBoostingClassifier(learning_rate=0.2,  n_estimators=100, max_depth=3, random_state=SEED)"),
        ]
        for tag, ctor in configs:
            cells.append(code(f"""\
m = {ctor}
mt = fit_eval(m, X_tr, y_tr, X_val, y_val)
print_m("{tag}", mt)
results_log.append({{'name': "GBM_{tag}", 'model': m, **mt}})
"""))
        cells.append(code(_RESULTS_PLOT_BLOCK("GradientBoosting")))
        return cells
    if model_name == "AdaBoost":
        intro = """\
## 4. Training - AdaBoost

AdaBoost re-weights mis-classified samples upward and then trains a new weak
learner. For binary classification SAMME.R is the default.

* Small `n_estimators` -> underfit (not enough boosting rounds).
* Large `n_estimators` -> overfit risk, especially on noisy targets like
  IEEE-CIS where the fraud labels themselves can be uncertain.
* The product `learning_rate x n_estimators` is the regularization knob.
* By default the base estimator is a stump (`max_depth=1`); we also test
  deeper bases (depth 3 and 5) which usually gives much better AUC.
"""
        body = """\
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)
"""
        cells = [md(intro), code(CV_HELPERS), code(body)]
        configs = [
            ("default n=50, lr=1.0",      "AdaBoostClassifier(n_estimators=50,  learning_rate=1.0, random_state=SEED)"),
            ("n=100, lr=1.0",             "AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=SEED)"),
            ("n=200, lr=0.5",             "AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED)"),
            ("n=400, lr=0.1",             "AdaBoostClassifier(n_estimators=400, learning_rate=0.1, random_state=SEED)"),
            ("base d=3, n=200, lr=0.5",   "AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED), n_estimators=200, learning_rate=0.5, random_state=SEED)"),
            ("base d=5, n=100, lr=1.0",   "AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5, random_state=SEED), n_estimators=100, learning_rate=1.0, random_state=SEED)"),
        ]
        for tag, ctor in configs:
            cells.append(code(f"""\
m = {ctor}
mt = fit_eval(m, X_tr, y_tr, X_val, y_val)
print_m("{tag}", mt)
results_log.append({{'name': "Ada_{tag}", 'model': m, **mt}})
"""))
        cells.append(code(_RESULTS_PLOT_BLOCK("AdaBoost")))
        return cells
    if model_name == "XGBoost":
        intro = """\
## 4. Training - XGBoost

XGBoost = optimized, regularized gradient boosting. It's the industry default
for IEEE-CIS and most public-leaderboard solutions are XGBoost-based.

Hyperparameters we sweep:

* `learning_rate` (eta), `n_estimators`, `max_depth` - the boosting capacity.
* `min_child_weight` - minimum sum of instance weights required to make a split
  (overfit knob).
* `subsample`, `colsample_bytree` - stochastic regularization (bagging at the
  row and column level).
* `scale_pos_weight` ~ N_neg / N_pos - the proper way to handle the fraud
  imbalance in XGBoost.
* `reg_alpha` (L1) and `reg_lambda` (L2) - on top of all of the above.
"""
        body = """\
import xgboost as xgb

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)

# Compute scale_pos_weight from training data
spw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
print(f"scale_pos_weight = {spw:.2f}")
"""
        cells = [md(intro), code(CV_HELPERS), code(body)]
        configs = [
            ("default-ish lr=0.1 n=300 d=6",    "xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, eval_metric='auc', tree_method='hist', random_state=SEED, n_jobs=-1)"),
            ("lr=0.05 n=600 d=8 spw",           "xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, scale_pos_weight=spw, eval_metric='auc', tree_method='hist', random_state=SEED, n_jobs=-1)"),
            ("lr=0.05 n=800 d=10 sub=0.8 cs=0.7","xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=10, subsample=0.8, colsample_bytree=0.7, scale_pos_weight=spw, eval_metric='auc', tree_method='hist', random_state=SEED, n_jobs=-1)"),
            ("lr=0.1 n=300 d=6 reg_l2=10",       "xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, reg_lambda=10, eval_metric='auc', tree_method='hist', random_state=SEED, n_jobs=-1)"),
            ("lr=0.03 n=1200 d=8 strict reg",    "xgb.XGBClassifier(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.7, eval_metric='auc', tree_method='hist', random_state=SEED, n_jobs=-1)"),
            ("shallow lr=0.1 n=500 d=4 reg",     "xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4, scale_pos_weight=spw, reg_alpha=0.5, reg_lambda=2.0, eval_metric='auc', tree_method='hist', random_state=SEED, n_jobs=-1)"),
        ]
        for tag, ctor in configs:
            cells.append(code(f"""\
m = {ctor}
mt = fit_eval(m, X_tr, y_tr, X_val, y_val)
print_m("{tag}", mt)
results_log.append({{'name': "XGB_{tag}", 'model': m, **mt}})
"""))
        cells.append(code(_RESULTS_PLOT_BLOCK("XGBoost")))
        return cells
    if model_name == "NeuralNetwork":
        intro = """\
## 4. Training - Neural Network (MLP via back-propagation)

A Multi-Layer Perceptron trained with back-prop on the log-loss. In real fraud
problems a small, well-regularized MLP usually trails XGBoost by a small margin
on AUC but adds value in an ensemble.

Hyperparameters we sweep:

* `hidden_layer_sizes` - architecture / capacity.
* `alpha` - L2 weight decay (regularization).
* `learning_rate_init`, `solver` (`adam` vs `sgd`).
* `early_stopping=True` so the optimizer stops before it starts memorizing.
"""
        body = """\
from sklearn.neural_network import MLPClassifier

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_sel = X_train_fe[SELECTED_FEATURES].astype(np.float32).values
X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.2,
                                            stratify=y, random_state=SEED)
scaler = StandardScaler()
X_tr_s, X_val_s = scaler.fit_transform(X_tr), scaler.transform(X_val)
print(f"X_tr {X_tr.shape}, X_val {X_val.shape}")
"""
        cells = [md(intro), code(CV_HELPERS), code(body)]
        configs = [
            ("(64,) alpha=1e-4 adam early-stop",   "MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-4, learning_rate_init=1e-3, solver='adam', early_stopping=True, max_iter=60, random_state=SEED)"),
            ("(128,64) alpha=1e-4 adam early-stop","MLPClassifier(hidden_layer_sizes=(128,64), alpha=1e-4, learning_rate_init=1e-3, solver='adam', early_stopping=True, max_iter=60, random_state=SEED)"),
            ("(256,128,64) alpha=1e-3 deep",       "MLPClassifier(hidden_layer_sizes=(256,128,64), alpha=1e-3, learning_rate_init=1e-3, solver='adam', early_stopping=True, max_iter=80, random_state=SEED)"),
            ("(128,) alpha=1e-2 strong reg",       "MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-2, learning_rate_init=1e-3, solver='adam', early_stopping=True, max_iter=60, random_state=SEED)"),
            ("(128,64) sgd lr=1e-2",               "MLPClassifier(hidden_layer_sizes=(128,64), alpha=1e-4, learning_rate_init=1e-2, solver='sgd', early_stopping=True, max_iter=60, random_state=SEED)"),
            ("wide (512,) alpha=1e-3",             "MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-3, learning_rate_init=1e-3, solver='adam', early_stopping=True, max_iter=60, random_state=SEED)"),
        ]
        for tag, ctor in configs:
            cells.append(code(f"""\
m = {ctor}
mt = fit_eval(m, X_tr_s, y_tr, X_val_s, y_val)
print_m("{tag}", mt)
results_log.append({{'name': "NN_{tag}", 'model': m, **mt}})
"""))
        cells.append(code(_RESULTS_PLOT_BLOCK("NeuralNetwork", scaled=True)))
        return cells
    raise ValueError(model_name)


def _RESULTS_PLOT_BLOCK(name, scaled=False):
    """Common results-summary cell at end of training section."""
    return f"""\
df_results = pd.DataFrame([{{k:v for k,v in r.items() if k!='model'}} for r in results_log])
df_results = df_results.sort_values('val_auc', ascending=False).reset_index(drop=True)

# Diagnose each run: overfit / underfit / healthy.
# Heuristic for fraud detection (AUC):
#   - train AUC < 0.75  -> underfit (model can't learn the task)
#   - overfit_gap > 0.05 -> overfit (memorising train, fails on val)
#   - 0 <= overfit_gap <= 0.02 and val_auc > 0.85 -> healthy
def _diag(row):
    if row['train_auc'] < 0.75:
        return 'UNDERFIT'
    if row['overfit_gap'] > 0.05:
        return 'OVERFIT'
    if row['overfit_gap'] < 0:
        return 'lucky-val'
    if row['val_auc'] >= 0.85 and row['overfit_gap'] <= 0.02:
        return 'HEALTHY'
    return 'mild-overfit' if row['overfit_gap'] > 0.02 else 'ok'
df_results['diagnosis'] = df_results.apply(_diag, axis=1)

_default_cols = ['name','train_auc','val_auc','val_f1','val_ap','overfit_gap','diagnosis']
show_cols = [c for c in _default_cols if c in df_results.columns]
print(df_results[show_cols].to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
df_results.set_index('name')[['train_auc','val_auc']].plot(kind='bar', ax=ax[0])
ax[0].set_title('{name}: Train vs Val AUC')
ax[0].set_ylim(max(0.5, df_results[['train_auc','val_auc']].min().min()-0.05), 1.0)
ax[0].tick_params(axis='x', rotation=30); ax[0].legend(loc='lower right')

colors = ['salmon' if g > 0.05 else ('orange' if g > 0.02 else 'seagreen')
          for g in df_results['overfit_gap']]
df_results.set_index('name')['overfit_gap'].plot(kind='bar', ax=ax[1], color=colors)
ax[1].axhline(0,    color='black', lw=0.5)
ax[1].axhline(0.02, color='orange', ls='--', lw=0.5, label='mild')
ax[1].axhline(0.05, color='red',    ls='--', lw=0.5, label='overfit')
ax[1].set_title('{name}: overfit gap (Train AUC - Val AUC)')
ax[1].tick_params(axis='x', rotation=30); ax[1].legend(loc='upper right')
plt.tight_layout(); plt.show()

n_over   = int((df_results['diagnosis']=='OVERFIT').sum())
n_under  = int((df_results['diagnosis']=='UNDERFIT').sum())
n_health = int((df_results['diagnosis']=='HEALTHY').sum())
best     = df_results.iloc[0]
worst    = df_results.iloc[-1]

print(f\"\"\"
=========== ANALYSIS ({name}) ===========
- {{len(df_results)}} runs total | HEALTHY: {{n_health}} | OVERFIT: {{n_over}} | UNDERFIT: {{n_under}}
- Best   : {{best['name']}}  ->  val AUC = {{best['val_auc']:.5f}},  gap = {{best['overfit_gap']:+.4f}}
- Worst  : {{worst['name']}}  ->  val AUC = {{worst['val_auc']:.5f}}, gap = {{worst['overfit_gap']:+.4f}}

How to read the diagnosis column:
- overfit_gap > 0.05  -> OVERFIT  (train AUC is much higher than val AUC, the model
                                   memorized the training data)
- overfit_gap <= 0.02 -> HEALTHY  (model learned the signal and generalizes)
- train_auc  < 0.75   -> UNDERFIT (high bias, model is too simple OR regularization
                                   is too strong)
- For fraud detection, recall (true-positive rate on the fraud class) is often
  prioritized alongside AUC. If the business wants high recall, the decision
  threshold should be moved below 0.5 - the AUC ranking does not change but the
  precision/recall trade-off does.
\"\"\")
best_model = [r['model'] for r in results_log if r['name']==best['name']][0]
print(f\"-> picked best_model = {{best['name']}}\")
"""


PIPELINE_BLOCK = """\
import pickle
from sklearn.pipeline import Pipeline

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.cols]

# Build a single pipeline that runs on RAW test data
final_pipeline = Pipeline([
    ('feat',   FeatureEngineer()),
    ('catenc', CategoricalEncoder()),
    ('impute', Imputer()),
    ('select', ColumnSelector(SELECTED_FEATURES)),
    ('model',  best_model),
])

# Refit FE part + best model on the FULL training data (raw)
final_pipeline.fit(X_train_raw, y)
print("Pipeline fitted on full raw training data.")

# Sanity: probabilistic predictions on raw test
test_pred_proba = final_pipeline.predict_proba(X_test_raw)[:, 1]
print(f"Test prediction probabilities sample: {test_pred_proba[:5]}")
print(f"Mean predicted P(fraud) on test set : {test_pred_proba.mean():.4f}")

# Save pipeline locally too (optional)
PIPE_PATH = f"pipeline_{MODEL_TAG}.pkl"
with open(PIPE_PATH, 'wb') as f:
    pickle.dump(final_pipeline, f)
print(f"Pipeline saved to {PIPE_PATH}")
"""

# MLflow logging cells (separate cells per run)
MLFLOW_BLOCK_TEMPLATE = """\
## 6. MLflow Logging (run separately!)

This section is split into separate cells so you can finish all the modelling
work first and only push runs to MLflow when you are ready.

Experiment name: **{exp_name}**.

The runs created here are:

* `<Model>_Cleaning` - dropped columns + before/after shapes
* `<Model>_Feature_Selection` - per-method AUC + chosen feature set
* `<Model>_<config>` - one run per hyperparameter combination from section 4
* `<Model>_CrossValidation` - 5-fold CV for the best config
* `<Model>_Final_Pipeline` - logs the complete sklearn `Pipeline` and registers
  it in the Model Registry under `IEEE_Fraud_<Model>`.
"""

MLFLOW_LOG_TRAINS = """\
# 6.1  Per-hyperparameter runs
mlflow.set_experiment(MLFLOW_EXPERIMENT)

for r in results_log:
    with mlflow.start_run(run_name=r['name']):
        # Params (model + general)
        mlflow.log_param('model_type',        MODEL_TAG)
        mlflow.log_param('n_features',        len(SELECTED_FEATURES))
        mlflow.log_param('feature_selection', best_fs_name if 'best_fs_name' in dir() else 'manual')
        mlflow.log_param('config',            r['name'])
        # Metrics
        for k, v in r.items():
            if k in ('name','model'): continue
            try: mlflow.log_metric(k, float(v))
            except Exception: pass
print("Logged all training runs to MLflow.")
"""

MLFLOW_LOG_FS = """\
# 6.2  Feature-Selection comparison run
with mlflow.start_run(run_name=f"{MODEL_TAG}_Feature_Selection"):
    mlflow.log_param('stage', 'feature_selection')
    mlflow.log_param('chosen', best_fs_name if 'best_fs_name' in dir() else 'n/a')
    mlflow.log_param('n_selected', len(SELECTED_FEATURES))
    if 'fs_df' in dir():
        for _, row in fs_df.iterrows():
            mlflow.log_metric(f"AUC_{row['method']}", float(row['mean_auc']))
print("Feature Selection run logged.")
"""

MLFLOW_LOG_CLEANING = """\
# 6.3  Cleaning summary run
with mlflow.start_run(run_name=f"{MODEL_TAG}_Cleaning"):
    mlflow.log_param('stage', 'cleaning')
    mlflow.log_param('dropped_columns', len(DROP_COLS))
    mlflow.log_param('train_shape_after', str(train.shape))
    mlflow.log_param('test_shape_after',  str(test.shape))
print("Cleaning run logged.")
"""

MLFLOW_LOG_CV = """\
# 6.4  Cross-validation run for the BEST hyperparameter set
print("Re-running 5-fold CV for the BEST config (this can take a few min)...")
cv5_aucs = cross_val_score(best_model, X_train_fe[SELECTED_FEATURES].values, y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                           scoring='roc_auc', n_jobs=-1)
print(f"CV AUC mean = {cv5_aucs.mean():.5f} +/- {cv5_aucs.std():.5f}")

with mlflow.start_run(run_name=f"{MODEL_TAG}_CrossValidation"):
    mlflow.log_param('stage', 'cross_validation')
    mlflow.log_param('cv_folds', 5)
    mlflow.log_param('best_config', best['name'])
    mlflow.log_metric('cv_auc_mean', float(cv5_aucs.mean()))
    mlflow.log_metric('cv_auc_std',  float(cv5_aucs.std()))
    for i, a in enumerate(cv5_aucs):
        mlflow.log_metric(f'cv_auc_fold{i+1}', float(a))
print("Cross-validation run logged.")
"""

MLFLOW_LOG_FINAL = """\
# 6.5  FINAL run -- log the trained Pipeline as MLflow artifact
with mlflow.start_run(run_name=f"{MODEL_TAG}_Final_Pipeline"):
    mlflow.log_param('model_type',        MODEL_TAG)
    mlflow.log_param('best_config',       best['name'])
    mlflow.log_param('n_features',        len(SELECTED_FEATURES))
    mlflow.log_param('feature_selection', best_fs_name if 'best_fs_name' in dir() else 'manual')

    mlflow.log_metric('best_val_auc',     float(best['val_auc']))
    mlflow.log_metric('best_train_auc',   float(best['train_auc']))
    mlflow.log_metric('best_overfit_gap', float(best['overfit_gap']))
    if 'cv5_aucs' in dir():
        mlflow.log_metric('cv_auc_mean', float(cv5_aucs.mean()))

    # Log entire pipeline (preprocessing + model) as an MLflow sklearn model
    # so model_inference can load it from the registry directly.
    mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        artifact_path='model',
        registered_model_name=f'IEEE_Fraud_{MODEL_TAG}',
    )

print(f"Final pipeline logged & registered as 'IEEE_Fraud_{MODEL_TAG}'.")
print("In model_inference.ipynb you can now load it via:")
print(f"    mlflow.sklearn.load_model('models:/IEEE_Fraud_{MODEL_TAG}/latest')")
"""


# ---------------------------------------------------------------------------
# Build a model_experiment_<name>.ipynb
# ---------------------------------------------------------------------------
def build_model_notebook(filename, model_name, model_kind, mlflow_exp):
    """model_kind in {linear, tree, nn}"""
    cells = [
        md(f"# IEEE-CIS Fraud Detection - {model_name}\n\n"
           "* **task**: binary classification (`isFraud`)  \n"
           "* **metric**: ROC-AUC  \n"
           "* **MLflow experiment**: `" + mlflow_exp + "`  \n"
           "* **runs in this experiment**:\n"
           f"   * `{model_name}_Cleaning`\n"
           f"   * `{model_name}_Feature_Selection`\n"
           f"   * `{model_name}_<config>` (one per hyperparam combo)\n"
           f"   * `{model_name}_CrossValidation`\n"
           f"   * `{model_name}_Final_Pipeline`  (logs the `Pipeline` to Model Registry)\n"
           "\nMLflow logging code is at the very bottom in **separate cells** "
           "so you can run modelling first and only log when ready."),
        md("## 0. Setup"),
        code(SETUP_IMPORTS),
        code(DAGSHUB_INIT),
        code(f"MODEL_TAG = '{model_name}'\n"
             f"MLFLOW_EXPERIMENT = '{mlflow_exp}'\n"
             f"print('MLflow experiment:', MLFLOW_EXPERIMENT)"),
        code(DATA_LOADER),
        code(DATA_OVERVIEW),
        md(CLEANING_INTRO),
        code(CLEANING_CODE),
        code(CLEANING_ANALYSIS),
        md(FE_INTRO),
        code(FE_TRANSFORMERS),
        code(FE_APPLY),
        code(FE_ANALYSIS_PLOT),
    ]
    cells.extend(feature_selection_section(model_kind))
    cells.extend(training_section(model_name))
    cells.append(md(PIPELINE_INTRO))
    cells.append(code(PIPELINE_BLOCK))
    cells.append(md(MLFLOW_BLOCK_TEMPLATE.format(exp_name=mlflow_exp)))
    cells.append(code(MLFLOW_LOG_CLEANING))
    cells.append(code(MLFLOW_LOG_FS))
    cells.append(code(MLFLOW_LOG_TRAINS))
    cells.append(code(MLFLOW_LOG_CV))
    cells.append(code(MLFLOW_LOG_FINAL))
    write_notebook(OUT_DIR / filename, cells)


# ---------------------------------------------------------------------------
# Inference notebook
# ---------------------------------------------------------------------------
INFERENCE_INTRO = """\
# IEEE-CIS Fraud Detection - Model Inference

This notebook:

1. Loads the **best model's Pipeline from the MLflow Model Registry** (a single
   sklearn `Pipeline`, so we don't have to repeat preprocessing here).
2. Reads the RAW `test_transaction.csv` and `test_identity.csv` and merges them.
3. Calls `pipeline.predict_proba(raw_X)`.
4. Writes a Kaggle-ready `submission.csv` (`TransactionID,isFraud`).

To switch which architecture is used, change `REGISTERED_NAME` below to whichever
`IEEE_Fraud_<Model>` has the highest CV AUC after running all experiment notebooks.
"""

INFERENCE_CODE = """\
import os, gc, pandas as pd, numpy as np
import mlflow, mlflow.sklearn, dagshub
import warnings; warnings.filterwarnings('ignore')

REPO_OWNER = "rkvit23"
REPO_NAME  = "ML-HW2"
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

# Pick the model to use. After all experiments are done you can change this
# to whichever architecture has the best CV AUC.
REGISTERED_NAME = "IEEE_Fraud_XGBoost"   # <-- change to the winning architecture
MODEL_URI       = f"models:/{REGISTERED_NAME}/latest"

print(f"Loading {MODEL_URI} from MLflow ...")
pipeline = mlflow.sklearn.load_model(MODEL_URI)
print("Loaded:", type(pipeline))
"""

INFERENCE_PREDICT = """\
DATA_DIR = "data"   # change to "/kaggle/input/ieee-fraud-detection" on Kaggle

print("Loading raw test data...")
test_tx = pd.read_csv(os.path.join(DATA_DIR, "test_transaction.csv"))
test_id = pd.read_csv(os.path.join(DATA_DIR, "test_identity.csv"))
test_id.columns = [c.replace('-', '_') for c in test_id.columns]
test = test_tx.merge(test_id, on="TransactionID", how="left")
del test_tx, test_id; gc.collect()

test_ids = test["TransactionID"].copy()
X_raw    = test.drop(columns=["TransactionID"])
print(f"Raw test shape: {X_raw.shape}")

# The pipeline includes ALL preprocessing -> preds in one call
print("Predicting...")
preds = pipeline.predict_proba(X_raw)[:, 1]
print(f"Done. mean P(fraud) = {preds.mean():.5f}, min={preds.min():.5f}, max={preds.max():.5f}")
"""

INFERENCE_SUBMIT = """\
submission = pd.DataFrame({"TransactionID": test_ids, "isFraud": preds})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved:", submission.shape)
submission.head()
"""

INFERENCE_PLOT = """\
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(preds, bins=80, color='steelblue', edgecolor='black')
ax.set_xlabel('Predicted P(isFraud)'); ax.set_ylabel('count')
ax.set_title('Predicted fraud-probability distribution (test set)')
plt.tight_layout(); plt.show()

print("Top-10 highest-risk transactions:")
print(submission.sort_values('isFraud', ascending=False).head(10).to_string(index=False))
"""

def build_inference_notebook():
    cells = [
        md(INFERENCE_INTRO),
        md("## 1. Load best Pipeline from MLflow Model Registry"),
        code(INFERENCE_CODE),
        md("## 2. Predict on raw test set"),
        code(INFERENCE_PREDICT),
        md("## 3. Generate `submission.csv`"),
        code(INFERENCE_SUBMIT),
        md("## 4. Sanity-check predictions"),
        code(INFERENCE_PLOT),
    ]
    write_notebook(OUT_DIR / "model_inference.ipynb", cells)


# ---------------------------------------------------------------------------
# Specs for all the model notebooks
# ---------------------------------------------------------------------------
SPECS = [
    ("model_experiment_LinearRegression.ipynb",   "LinearRegression",   "linear", "LinearRegression_Training"),
    ("model_experiment_LogisticRegression.ipynb", "LogisticRegression", "linear", "LogisticRegression_Training"),
    ("model_experiment_GLM.ipynb",                "GLM",                "linear", "GLM_Training"),
    ("model_experiment_DecisionTree.ipynb",       "DecisionTree",       "tree",   "DecisionTree_Training"),
    ("model_experiment_Bagging.ipynb",            "Bagging",            "tree",   "Bagging_Training"),
    ("model_experiment_RandomForest.ipynb",       "RandomForest",       "tree",   "RandomForest_Training"),
    ("model_experiment_GradientBoosting.ipynb",   "GradientBoosting",   "tree",   "GradientBoosting_Training"),
    ("model_experiment_AdaBoost.ipynb",           "AdaBoost",           "tree",   "AdaBoost_Training"),
    ("model_experiment_XGBoost.ipynb",            "XGBoost",            "tree",   "XGBoost_Training"),
    ("model_experiment_NeuralNetwork.ipynb",      "NeuralNetwork",      "nn",     "NeuralNetwork_Training"),
]

if __name__ == "__main__":
    print("Building model_experiment notebooks...")
    for filename, model_name, kind, exp in SPECS:
        build_model_notebook(filename, model_name, kind, exp)
    print("\nBuilding model_inference notebook...")
    build_inference_notebook()
    print("\nAll notebooks generated.")
