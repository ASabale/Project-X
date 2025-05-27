# Step 1: Set up environment and imports

import numpy as np
import pandas as pd
import random
import os

# For model training
import xgboost as xgb
import lightgbm as lgb

# For preprocessing
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split

# For SMOTE
from imblearn.over_sampling import SMOTE

# For metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# For saving artifacts
import joblib

# Set global seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Set seeds for frameworks
xgb.random.seed(SEED)
lgb_params = {'random_state': SEED}  # Pass in LightGBM training

print("Environment and imports are ready. Seed set to:", SEED)

import pandas as pd
df = pd.read_csv('drive/MyDrive/Colab Notebooks/data.csv')

# Split data by month ---
available_df = df[df['month'].between(0, 5)].reset_index(drop=True)
future_df = df[df['month'].isin([6, 7])].reset_index(drop=True)

# Assign each set by month
train_df = available_df[available_df['month'].between(0, 3)].reset_index(drop=True)
val_df   = available_df[available_df['month'] == 4].reset_index(drop=True)
devtest_df = available_df[available_df['month'] == 5].reset_index(drop=True)

print(f"Train set:     {train_df.shape}")
print(f"Validation set:{val_df.shape}")
print(f"Dev-test set:  {devtest_df.shape}")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Columns
cols_to_drop = ['proposed_credit_limit']
cols_to_transform = [
    'prev_address_months_count',
    'current_address_months_count',
    'days_since_request',
    'intended_balcon_amount',
    'zip_count_4w',
    'velocity_4w',
    'bank_branch_count_8w',
    'session_length_in_minutes'
]
categorical_cols = [
    'payment_type',
    'employment_status',
    'housing_status',
    'source',
    'device_os'
]
label_col = 'fraud_bool'

# 2. Drop column, split features/labels
def prepare_xy(df):
    X = df.drop(cols_to_drop + [label_col], axis=1)
    y = df[label_col].copy()
    return X, y

X_train, y_train = prepare_xy(train_df)
X_val, y_val = prepare_xy(val_df)
X_dev, y_dev = prepare_xy(devtest_df)

# 3. Preprocessing pipeline
# Yeo-Johnson for skewed columns, OneHot for categoricals, passthrough for rest
preprocessor = ColumnTransformer([
    ('yeojohnson', PowerTransformer(method='yeo-johnson'), cols_to_transform),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
], remainder='passthrough')

# 4. Fit only on training data
preprocessor.fit(X_train)

# Save the transformer for later use (future_df)
joblib.dump(preprocessor, 'preprocessor.pkl')

# 5. Transform all sets
X_train_prep = preprocessor.transform(X_train)
X_val_prep   = preprocessor.transform(X_val)
X_dev_prep   = preprocessor.transform(X_dev)

print("Preprocessing complete.")
print("Transformed train shape:", X_train_prep.shape)
print("Transformed val shape:  ", X_val_prep.shape)
print("Transformed dev-test shape:", X_dev_prep.shape)

from imblearn.over_sampling import SMOTE

# Set up SMOTE
smote = SMOTE(random_state=SEED)

# Fit and resample only the training data
X_train_bal, y_train_bal = smote.fit_resample(X_train_prep, y_train)

print("After SMOTE:")
print("Balanced train shape:", X_train_bal.shape)
print("Class distribution:", np.bincount(y_train_bal))
joblib.dump(smote, 'smote.pkl')

import xgboost as xgb

# XGBoost DMatrix
dtrain = xgb.DMatrix(X_train_bal, label=y_train_bal)
dval   = xgb.DMatrix(X_val_prep, label=y_val)

# Parameters (tune further as needed)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'tree_method': 'hist',      # Use 'hist'
    'device': 'cuda',           # Add this for GPU
    'random_state': SEED,
    'scale_pos_weight': 1,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}


evals = [(dtrain, 'train'), (dval, 'val')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=100
)

# Save best model
bst.save_model('xgb_model.json')

import lightgbm as lgb

# LightGBM Dataset
lgb_train = lgb.Dataset(X_train_bal, label=y_train_bal)
lgb_val = lgb.Dataset(X_val_prep, label=y_val, reference=lgb_train)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',
    'random_state': SEED,
    'boosting_type': 'gbdt',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
}

import lightgbm as lgb

lgbm = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)


# Save best model
lgbm.save_model('lgbm_model.txt')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score

# ---- XGBoost: Predict probabilities on val ----
dval = xgb.DMatrix(X_val_prep)
xgb_val_pred = bst.predict(dval)

# ---- LightGBM: Predict probabilities on val ----
lgb_val_pred = lgbm.predict(X_val_prep, num_iteration=lgbm.best_iteration)

# ---- ROC Curve ----
def plot_roc(y_true, pred_probs, label):
    fpr, tpr, _ = roc_curve(y_true, pred_probs)
    auc_val = roc_auc_score(y_true, pred_probs)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_val:.3f})')

plt.figure(figsize=(8,6))
plot_roc(y_val, xgb_val_pred, 'XGBoost')
plot_roc(y_val, lgb_val_pred, 'LightGBM')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation Set)')
plt.legend()
plt.show()

# ---- Confusion Matrix ----
def plot_conf_matrix(y_true, preds, model_label, threshold=0.5):
    y_pred = (preds >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{model_label} Confusion Matrix (thresh={threshold}):\n", cm)
    print(classification_report(y_true, y_pred, digits=4))

plot_conf_matrix(y_val, xgb_val_pred, "XGBoost")
plot_conf_matrix(y_val, lgb_val_pred, "LightGBM")

# ---- Precision-Recall Curve (Optional but recommended) ----
def plot_pr_curve(y_true, pred_probs, label):
    precision, recall, _ = precision_recall_curve(y_true, pred_probs)
    ap = average_precision_score(y_true, pred_probs)
    plt.plot(recall, precision, label=f'{label} (AP = {ap:.3f})')

plt.figure(figsize=(8,6))
plot_pr_curve(y_val, xgb_val_pred, 'XGBoost')
plot_pr_curve(y_val, lgb_val_pred, 'LightGBM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation Set)')
plt.legend()
plt.show()