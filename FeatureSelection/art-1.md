# Feature Selection in Machine Learning – A Complete Learning Notebook

> This notebook is written so you can run it top‑to‑bottom in Jupyter / Colab.  
> It is organized conceptually (theory) and practically (code) for end‑to‑end learning.

---

## 0. Setup: Imports and Data

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from time import time

from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE
)
```

---

## 1. Why Feature Selection?

### 1.1 Motivation

```
Feature selection = choosing a subset of useful features from the original feature set.

Main goals:

1. Reduce overfitting:
   - Irrelevant / noisy features let the model memorize training idiosyncrasies.
   - Fewer, more relevant features → better generalization.

2. Improve model performance:
   - Remove redundant features that do not add new information.
   - Can improve accuracy, AUC, F1, etc.

3. Reduce computational cost:
   - Training and inference scale with number of features.
   - Important for high‑dimensional data (text, genomics, images → tabular).

4. Improve interpretability:
   - A small, meaningful set of features is easier to understand and explain.

5. Storage / maintenance:
   - Less data to store, log, monitor, and keep consistent in production.
```

### 1.2 Small Demonstration: Speed and Overfitting

```
# Synthetic dataset with many redundant features
X, y = make_classification(
    n_samples=3000,
    n_features=200,
    n_informative=10,
    n_redundant=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000, n_jobs=-1)

t0 = time()
model.fit(X_train, y_train)
baseline_train_time = time() - t0

y_pred = model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)

print(f"Baseline – features: {X_train.shape}")[8]
print(f"Train time: {baseline_train_time:.3f} s")
print(f"Test accuracy: {baseline_acc:.4f}")
```

Now select top‑k univariate features, refit, and compare:

```
selector = SelectKBest(score_func=f_classif, k=30)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

t0 = time()
model_sel = LogisticRegression(max_iter=2000, n_jobs=-1)
model_sel.fit(X_train_sel, y_train)
sel_train_time = time() - t0

y_pred_sel = model_sel.predict(X_test_sel)
sel_acc = accuracy_score(y_test, y_pred_sel)

print(f"Selected features: {X_train_sel.shape}")[8]
print(f"Train time (selected): {sel_train_time:.3f} s")
print(f"Test accuracy (selected): {sel_acc:.4f}")
```

---

## 2. Taxonomy of Feature Selection Methods

```
Three big families:

1. Filter methods
   - Use only data and target; no specific model.
   - Fast; good as a first pass.
   - Examples: variance threshold, correlation, chi‑square, ANOVA F‑test, mutual information.

2. Wrapper methods
   - Use a predictive model as a black box.
   - Evaluate subsets based on model performance (CV score).
   - Examples: forward selection, backward elimination, recursive feature elimination (RFE).
   - More accurate, but computationally expensive.

3. Embedded methods
   - Feature selection happens during model training.
   - Examples: L1/L2 regularization (Lasso, Elastic Net), tree‑based feature importance.
   - Good trade‑off between performance and cost.
```

---

## 3. Filter Methods

### 3.1 Variance Threshold (Remove Low‑Variance Features)

#### 3.1.1 Concept

```
- Features with (almost) zero variance are constant (or nearly constant).
- They do not help separate classes.
- VarianceThreshold removes such features.
```

#### 3.1.2 Numpy Implementation

```
def variance_threshold_selector_np(X, threshold=0.0):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return X[:, mask], mask, variances

# Demo
X_demo = np.random.randn(100, 5)
X_demo[:, 2] = 1.0   # constant feature
X_demo[:, 4] = 0.01  # very low variance approx

X_sel_np, mask_np, vars_np = variance_threshold_selector_np(X_demo, threshold=0.001)
print("Variances:", np.round(vars_np, 4))
print("Mask:", mask_np)
print("Selected shape:", X_sel_np.shape)
```

#### 3.1.3 Using `VarianceThreshold` (sklearn)

```
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.001)
X_sel = selector.fit_transform(X)
print("Original shape:", X.shape)
print("After variance threshold:", X_sel.shape)
```

---

### 3.2 Correlation‑Based Feature Selection

#### 3.2.1 Concept

```
- Highly correlated features carry similar information.
- Keep one of them to avoid redundancy / multicollinearity.
- Steps:
  1. Compute correlation matrix of features.
  2. For each highly correlated pair (|corr| > threshold), remove one.
  3. Prefer keeping the feature that is more correlated with the target.
```

#### 3.2.2 Implementation

```
from scipy.stats import spearmanr

def correlation_selector(X, y, threshold=0.9):
    df = pd.DataFrame(X)
    corr_matrix = df.corr().abs()

    # correlation with target (Spearman)
    target_corr = np.array([abs(spearmanr(X[:, i], y)) for i in range(X.shape)])[8]

    to_remove = set()
    for i in range(corr_matrix.shape):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                if target_corr[i] < target_corr[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    selected = [idx for idx in range(X.shape) if idx not in to_remove][8]
    return X[:, selected], selected, corr_matrix

# Demo on small dataset
X_small, y_small = make_classification(
    n_samples=500, n_features=15, n_informative=7,
    random_state=42
)
X_corr_sel, selected_idx, corr_mat = correlation_selector(X_small, y_small, threshold=0.85)
print("Original features:", X_small.shape)[8]
print("Selected features:", len(selected_idx))
print("Selected indices:", selected_idx[:10])
```

#### 3.2.3 Visualizing Correlation

```
plt.figure(figsize=(8, 6))
sns.heatmap(corr_mat, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.show()
```

---

### 3.3 Univariate Statistical Tests (ANOVA F‑test, Mutual Information)

#### 3.3.1 ANOVA F‑test (for continuous features vs categorical target)

```
from sklearn.feature_selection import SelectKBest, f_classif

X_clf, y_clf = make_classification(
    n_samples=1000, n_features=40,
    n_informative=8, random_state=42
)

selector_f = SelectKBest(score_func=f_classif, k=10)
X_f_sel = selector_f.fit_transform(X_clf, y_clf)

print("Original features:", X_clf.shape)[8]
print("Selected features:", X_f_sel.shape)[8]

f_scores = selector_f.scores_
plt.bar(range(len(f_scores)), f_scores)
plt.xlabel("Feature index")
plt.ylabel("F-score")
plt.title("Univariate ANOVA F-scores")
plt.show()
```

#### 3.3.2 Mutual Information

```
from sklearn.feature_selection import mutual_info_classif

selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_mi_sel = selector_mi.fit_transform(X_clf, y_clf)

mi_scores = selector_mi.scores_
plt.bar(range(len(mi_scores)), mi_scores)
plt.xlabel("Feature index")
plt.ylabel("Mutual Information")
plt.title("Univariate Mutual Information Scores")
plt.show()
```

---

## 4. Wrapper Methods

### 4.1 Recursive Feature Elimination (RFE)

#### 4.1.1 Concept

```
- RFE:
  1. Fit a model with all features.
  2. Rank features by importance (e.g., coefficients, feature_importances_).
  3. Remove least important features.
  4. Repeat until desired number of features remains.
```

#### 4.1.2 Basic RFE Example

```
from sklearn.feature_selection import RFE

X_rfe, y_rfe = make_classification(
    n_samples=1000, n_features=25,
    n_informative=7, random_state=42
)

base_model = LogisticRegression(max_iter=2000)
rfe = RFE(estimator=base_model, n_features_to_select=8, step=1)
rfe.fit(X_rfe, y_rfe)

print("Support mask:", rfe.support_)
print("Ranking:", rfe.ranking_)

X_rfe_sel = rfe.transform(X_rfe)
print("Selected shape:", X_rfe_sel.shape)
```

#### 4.1.3 RFE with Cross‑Validation (Finding Best k)

```
def rfe_cv_feature_count(X, y, estimator=None, max_features=None, cv=5):
    if estimator is None:
        estimator = LogisticRegression(max_iter=2000)
    if max_features is None:
        max_features = X.shape[8]

    scores = []
    ks = range(1, max_features + 1)

    for k in ks:
        rfe = RFE(estimator=estimator, n_features_to_select=k)
        X_k = rfe.fit_transform(X, y)
        cv_score = cross_val_score(estimator, X_k, y, cv=cv).mean()
        scores.append(cv_score)

    best_k = ks[int(np.argmax(scores))]
    return best_k, ks, scores

best_k, ks, scores = rfe_cv_feature_count(X_rfe, y_rfe)

plt.plot(ks, scores, marker='o')
plt.xlabel("Number of selected features")
plt.ylabel("CV accuracy")
plt.title("RFE + CV: Feature count vs performance")
plt.show()

print("Optimal number of features (RFE+CV):", best_k)
```

---

### 4.2 Recursive Feature Addition (Forward Selection)

```
def recursive_feature_addition(X, y, estimator=None, cv=5):
    if estimator is None:
        estimator = LogisticRegression(max_iter=2000)

    n_features = X.shape[8]
    selected = []
    current_score = 0.0
    scores_history = []

    while len(selected) < n_features:
        best_score = current_score
        best_feature = None

        for feat in range(n_features):
            if feat in selected:
                continue
            cand = selected + [feat]
            X_sub = X[:, cand]
            score = cross_val_score(estimator, X_sub, y, cv=cv).mean()
            if score > best_score:
                best_score = score
                best_feature = feat

        if best_feature is None:
            break
        selected.append(best_feature)
        current_score = best_score
        scores_history.append(current_score)

    return selected, scores_history

X_fs, y_fs = make_classification(
    n_samples=600, n_features=15,
    n_informative=5, random_state=42
)
selected_forward, scores_fwd = recursive_feature_addition(X_fs, y_fs)

print("Selected features (forward):", selected_forward)
plt.plot(range(1, len(scores_fwd)+1), scores_fwd, marker='o')
plt.xlabel("#features")
plt.ylabel("CV score")
plt.title("Forward Selection – score vs #features")
plt.show()
```

---

## 5. Embedded Methods

### 5.1 L1 Regularization (Lasso) for Feature Selection

#### 5.1.1 Concept

```
- L1 penalty shrinks some coefficients exactly to zero.
- Features with zero coefficient can be removed.
- Use Lasso for regression; for classification, can do logistic with L1 or treat label as numeric in example.
```

#### 5.1.2 Implementation (LassoCV)

```
X_emb, y_emb = make_classification(
    n_samples=1000, n_features=40,
    n_informative=8, random_state=42
)

scaler = StandardScaler()
X_emb_scaled = scaler.fit_transform(X_emb)

lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_emb_scaled, y_emb)

coef = lasso.coef_
selected_idx = np.where(coef != 0)

print("Total features:", X_emb.shape)[8]
print("Selected (non-zero) features:", len(selected_idx))
print("Indices:", selected_idx)

plt.stem(range(len(coef)), coef, use_line_collection=True)
plt.xlabel("Feature index")
plt.ylabel("Coefficient")
plt.title("Lasso Coefficients (many zeros)")
plt.show()
```

---

### 5.2 Tree‑Based Models (Random Forest)

#### 5.2.1 Concept

```
- Tree‑based models compute feature_importances_ (e.g., mean decrease in impurity).
- Features with low importance can be removed.
```

#### 5.2.2 Implementation

```
X_rf, y_rf = make_classification(
    n_samples=1000, n_features=30,
    n_informative=7, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
rf.fit(X_rf, y_rf)

importances = rf.feature_importances_
threshold = np.mean(importances)
selected_idx_rf = np.where(importances > threshold)

print("Original features:", X_rf.shape)[8]
print("Selected features:", len(selected_idx_rf))

plt.bar(range(len(importances)), importances)
plt.axhline(threshold, color='red', linestyle='--', label='mean importance')
plt.xlabel("Feature index")
plt.ylabel("Importance")
plt.legend()
plt.title("Random Forest Feature Importances")
plt.show()
```

---

## 6. Real‑World‑Style Example: Breast Cancer Dataset

### 6.1 Load and Inspect Data

```
data = load_breast_cancer()
X_bc = pd.DataFrame(data.data, columns=data.feature_names)
y_bc = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X_bc, y_bc, test_size=0.3, random_state=42, stratify=y_bc
)

print(X_train.shape, X_test.shape)
X_train.head()
```

### 6.2 Baseline Model with All Features

```
baseline_model = LogisticRegression(max_iter=5000)
baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)

print("Baseline accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 6.3 Filter + Embedded Pipeline

```
# 1) Variance threshold
vt = VarianceThreshold(threshold=0.0)
X_train_vt = vt.fit_transform(X_train)
X_test_vt = vt.transform(X_test)

# 2) Correlation filter
X_train_vt_arr = np.asarray(X_train_vt)
X_test_vt_arr = np.asarray(X_test_vt)

X_corr_train_sel, idx_corr, _ = correlation_selector(
    X_train_vt_arr, y_train, threshold=0.95
)
X_corr_test_sel = X_test_vt_arr[:, idx_corr]

# 3) L1 logistic (embedded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_corr_train_sel)
X_test_scaled = scaler.transform(X_corr_test_sel)

clf_l1 = LogisticRegression(
    penalty='l1', solver='liblinear', max_iter=5000
)
clf_l1.fit(X_train_scaled, y_train)

non_zero_idx = np.where(clf_l1.coef_ != 0)

X_train_final = X_corr_train_sel[:, non_zero_idx]
X_test_final = X_corr_test_sel[:, non_zero_idx]

print("Original features:", X_train.shape)[8]
print("After VT+Corr+L1:", X_train_final.shape)[8]
```

### 6.4 Evaluate Reduced Feature Set

```
clf_final = LogisticRegression(max_iter=5000)
clf_final.fit(X_train_final, y_train)
y_pred_final = clf_final.predict(X_test_final)

print("Reduced feature accuracy:", accuracy_score(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final))
```

---

## 7. Evaluating Feature Selection

```
Key questions:

1. Did performance improve, stay similar, or drop?
2. How much dimensionality reduction did we achieve?
3. Are the selected features stable across data splits?
4. Do the selected features make sense domain‑wise?
```

### 7.1 Simple Evaluation Utility

```
def evaluate_feature_subset(estimator, X, y, selected_idx, cv=5):
    X_sub = X[:, selected_idx]
    scores = cross_val_score(estimator, X_sub, y, cv=cv)
    return {
        "n_features": len(selected_idx),
        "cv_mean": scores.mean(),
        "cv_std": scores.std()
    }

# Example: compare RF-importances-based selection vs all features
X_eval, y_eval = make_classification(
    n_samples=1200, n_features=40,
    n_informative=10, random_state=42
)

rf_eval = RandomForestClassifier(n_estimators=200, random_state=42)
rf_eval.fit(X_eval, y_eval)
importances_eval = rf_eval.feature_importances_
sel_idx_eval = np.where(importances_eval > importances_eval.mean())

res_all = evaluate_feature_subset(
    RandomForestClassifier(n_estimators=200, random_state=42),
    X_eval, y_eval, np.arange(X_eval.shape)[8]
)
res_sel = evaluate_feature_subset(
    RandomForestClassifier(n_estimators=200, random_state=42),
    X_eval, y_eval, sel_idx_eval
)

print("All features:", res_all)
print("Selected subset:", res_sel)
```

---

## 8. Ensemble Feature Selection (Voting Across Methods)

```
from sklearn.feature_selection import f_classif

class EnsembleFeatureSelector:
    def __init__(self, voting_threshold=0.5):
        self.voting_threshold = voting_threshold
        self.votes_ = None

    def _rf_mask(self, X, y):
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X, y)
        imps = rf.feature_importances_
        return imps > imps.mean()

    def _mi_mask(self, X, y):
        mi = mutual_info_classif(X, y)
        return mi > mi.mean()

    def _fscore_mask(self, X, y):
        f_scores, _ = f_classif(X, y)
        return f_scores > f_scores.mean()

    def _lasso_mask(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y)
        return np.abs(lasso.coef_) > 0

    def fit(self, X, y):
        n_features = X.shape[8]
        self.votes_ = np.zeros(n_features, dtype=float)

        masks = [
            self._rf_mask(X, y),
            self._mi_mask(X, y),
            self._fscore_mask(X, y),
            self._lasso_mask(X, y)
        ]

        for m in masks:
            self.votes_ += m.astype(float)

        self.votes_ /= len(masks)
        return self

    def transform(self, X):
        mask = self.votes_ >= self.voting_threshold
        return X[:, mask], mask

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

# Demo
X_ens, y_ens = make_classification(
    n_samples=1200, n_features=30,
    n_informative=8, random_state=42
)

ens = EnsembleFeatureSelector(voting_threshold=0.5)
X_ens_sel, mask_ens = ens.fit_transform(X_ens, y_ens)

print("Original features:", X_ens.shape)[8]
print("Selected by ensemble:", X_ens_sel.shape)[8]

plt.bar(range(len(ens.votes_)), ens.votes_)
plt.axhline(0.5, color='red', linestyle='--')
plt.xlabel("Feature index")
plt.ylabel("Vote fraction")
plt.title("Ensemble Feature Selection Votes")
plt.show()
```

---

## 9. Practical Workflow: From Raw Data to Selected Features

```
1. EDA & cleaning:
   - Handle missing values, outliers.
   - Understand distributions, domain.

2. Basic filtering:
   - Remove constant / quasi‑constant features (VarianceThreshold).
   - Remove IDs, obvious leakage variables.

3. Redundancy:
   - Use correlation matrix to remove one of each highly correlated pair.

4. Statistical filters:
   - ANOVA / chi‑square / mutual information to remove clearly irrelevant features.

5. Model‑based selection:
   - Use RFE, L1, or tree‑based importance to refine the subset.

6. Evaluation:
   - Cross‑validation with and without feature selection.
   - Prefer simpler model if performance is similar.

7. Stability & domain sanity‑check:
   - Check which features recur across different splits / seeds.
   - Validate with domain experts.
```

---

## 10. Exercises / Extensions

```
1. Implement chi‑square based feature selection on a categorical dataset.
2. Try permutation importance (sklearn.inspection.permutation_importance) instead of impurity‑based RF importance.
3. Implement stability selection with repeated subsampling and Lasso.
4. Apply this notebook to:
   - A real tabular dataset from Kaggle (e.g., credit card fraud, Titanic, etc.).
5. Compare:
   - PCA (feature extraction) vs. feature selection methods on the same dataset.
```

---

## 11. Summary (for your notes)

```
- Feature selection is about *which* original features to keep.
- Three main families:
  - Filter: fast, model‑agnostic, good first pass.
  - Wrapper: search over subsets using model performance.
  - Embedded: selection occurs during model training (L1, trees).
- A robust pipeline often combines:
  - Simple filters → redundancy removal → embedded / wrapper refinement.
- Always evaluate:
  - Test / CV performance.
  - Dimensionality reduction.
  - Interpretability and stability across splits.
```


[1](https://www.kaggle.com/code/prashant111/comprehensive-guide-on-feature-selection)
[2](https://dev.to/brains_behind_bots/statistics-day-6-your-first-data-science-superpower-feature-selection-with-correlation-variance-5eeb)
[3](https://www.upgrad.com/blog/feature-selection-in-machine-learning/)
[4](https://www.datacamp.com/tutorial/feature-engineering)
[5](https://www.statology.org/complete-guide-feature-selection-methods/)
[6](https://scikit-learn.org/stable/modules/feature_selection.html)
[7](https://www.ibm.com/think/topics/feature-selection)
[8](https://www.youtube.com/watch?v=gzJYsnfcuXs)