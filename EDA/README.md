# BAF Fraud Detection: EDA & Pre‑processing Pipeline

## Dataset Snapshot

* **Size:** \~1 million application records × **32 features**
* **Target:** `fraud_bool` (≈ 1 % positive class → **severely imbalanced**)
* **Feature types:**
  • 20 numeric
  • 8 categorical
  • 4 binary/flag

During initial inspection we:

1. **Counted unique values** for every feature to spot constants and high‑cardinality fields.
2. **Calculated fraud distribution** (counts + percent) and displayed a pie chart to visualise the imbalance.

## 1  Exploratory Data Analysis (EDA)

| Focus                             | What we did                                                                     | Why                                        |
| --------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------ |
| **Class imbalance**               | Pie chart of `fraud_bool` counts/%.                                             | Highlights need for resampling techniques. |
| **Unique counts**                 | Quick `.nunique()` overview.                                                    | Detect constant or quasi‑constant columns. |
| **Distributions by fraud status** | Histograms for numeric & count‑plots for categorical features (split by fraud). | Surfaces potential predictive patterns.    |

## 2  Missing‑Value Investigation

* Negative placeholders (`-1`, `-16`, *all negatives* for `intended_balcon_amount`) denote **missingness**.
* Calculated **% missing** per feature (see bar chart). 74 % of `intended_balcon_amount` & 71 % of `prev_address_months_count` are missing.
* Plotted **(a)** distributions using *all rows*, **(b)** distributions after **removing** missing rows, **(c)** fraud distribution **inside missing rows** → Missingness itself correlated with fraud.

### Decision

**Retain rows**. For each feature with a custom missing code we:

1. Created an indicator column `feature_missing` (1 if missing, else 0).
2. Imputed the missing numeric value with the **median** of observed data.

## 3  Correlation Analysis

* **Before** handling missing values → baseline correlation matrix.
* **After** indicator + imputation → second matrix (clearer structure, fewer spurious correlations).

## 4  Normalising Skewed Features

We benchmarked **Log**, **Box‑Cox**, and **Yeo‑Johnson** on eight highly skewed columns:

```python
cols_to_transform = [
    'prev_address_months_count','current_address_months_count','days_since_request',
    'intended_balcon_amount','zip_count_4w','velocity_4w',
    'bank_branch_count_8w','session_length_in_minutes']
```

**Yeo‑Johnson** improved normality without needing positive‑only data → selected.

* Stored transformer (`yeojohnson_transformer.pkl`) for **real‑time inference**.
* Re‑plotted distributions *before vs after* and recomputed correlations to verify improvement.

## 5  Categorical Feature Engineering

* Applied **one‑hot encoding** to 5 categorical columns (`payment_type`, `employment_status`, `housing_status`, `device_os`, `source`).
* Saved column list (`ohe_columns.pkl`) to guarantee consistent schema online.

## 6  Removing Low‑Information Features

* Dropped 6 constant / near‑constant columns uncovered via `.nunique()==1`.

## 7  Class Imbalance Solutions

| Method                | Purpose                                | Speed         | Info retention |
| --------------------- | -------------------------------------- | ------------- | -------------- |
| **Tomek Links**       | Clean class overlap                    | Fast          | High           |
| **Cluster Centroids** | Summarise majority via K‑Means         | Moderate      | Moderate       |
| **NearMiss‑1**        | Keep majority samples closest to fraud | Moderate/Slow | Moderate       |
| **ENN**               | Remove noisy border points             | Slowest       | High           |

> **Finding:** Tomek Links + SMOTE‑Tomek (future work) gave the best balance of recall and retained information. Random undersampling kept as baseline.

## 8  Feature Importance

* Trained **XGBoost** on the processed dataset (after Tomek Links).
* Generated bar plot of top‑20 features (`xgb.plot_importance` & seaborn barplot).
* Missing‑value indicators surfaced in top list → confirms earlier insight.
* **Temporal features** (e.g., `month`, `days_since_request`) ranked highly, validating datasheet note.

## 9  Scaling for Deep‑Learning Trials

* Plan to **standard‑scale** numeric features (post‑YJ) before feeding into DL models (tab‑transformers / autoencoders).

## Key Takeaways

* **Missingness ≠ noise**—encoding it increased model signal.
* Yeo‑Johnson normalisation improved correlations and stabilised tree‑model splits.
* Combining **cleansing (Tomek)** with **synthetic minority oversampling (SMOTE)** will likely boost recall for ML models.
* Proper encoding & saved artefacts (transformer, OHE columns) ensure pipeline consistency from batch training to real‑time inference.

## Next Steps

1. Integrate **SMOTE‑Tomek** pipeline and compare with NearMiss variants.
2. Hyper‑tune XGBoost & LightGBM; explore TabNet / FT‑Transformer with scaled inputs.
3. Deploy Spark + Kafka consumer to apply saved transformers and predict fraud on streaming events.

## Acknowledgements

* Inspired by [this kaggle notebook](https://www.kaggle.com/code/matthewmcnulty/bank-account-fraud#1.-Exploratory-Data-Analysis-of-Bank-Account-Applications).
* Thanks to the BAF team for providing the dataset and domain insights.