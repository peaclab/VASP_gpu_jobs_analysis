# usage: python3 runtime_power_pred.py lammps LGB run_name

# ***************
# Standard library imports
import os
import sys
import argparse
import pickle
import warnings
# ***************

# ***************
# Data science / numerical
import numpy as np
import pandas as pd
# ***************

# ***************
# Scikit-learn
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
# ***************

# ***************
# Gradient boosting libraries
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping as XGBEarlyStopping
# ***************

# ***************
# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
# ***************

# ***************
# Weights & Biases
import wandb
from wandb.integration.xgboost import WandbCallback
from wandb.keras import WandbCallback as KerasWandbCallback
# ***************

# ***************
# Custom functions
from metric_analysis import check_metrics, clean_and_compare
from prep_sequences import prepare_multivariate_sequences 
# ***************

warnings.filterwarnings('ignore', category=DeprecationWarning)


def bin_power(y):
    return pd.cut(
        y.ravel(),
        bins=[-np.inf, 100, 150, 200, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)
        

def train_max_baseline(y_test, window=3,
                       dataset_name='vasp',
                       wandb_run_name="Max_baseline_run"):
    """
    Max baseline: predicts next power class using the
    MAX of past power values in y_test.
    """
    print("Training Max baseline...")
    wandb.init(project="runtime_power_pred", name=wandb_run_name, reinit=True)

    print("y_test shape before:", np.asarray(y_test).shape)
    print("y_test shape after :", np.asarray(y_test).ravel().shape)

    y_test = np.asarray(y_test).astype(float).ravel()

    # actual usage
    y_true_cont = y_test[window:].reshape(-1, 1)
    y_true = bin_power(y_true_cont)

    # max over past window
    y_pred_cont = np.array([
        y_test[i:i+window].max()
        for i in range(len(y_test) - window)
    ]).reshape(-1, 1)

    y_pred = bin_power(y_pred_cont)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    print("\n***** Max Baseline Results *****")
    print(f"Accuracy: {acc:.4f}")
    print(cm)
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1, 2, 3],
        zero_division=0
    ))

    wandb.log({
        "max_baseline_accuracy": acc,
        "max_baseline_confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=["0", "1", "2", "3"]
        )
    })

    model_dict = {
        'model': None,
        'X_test': None,
        'y_test': y_test,
        'y_train_cls':y_true,
        'y_pred': y_pred,
        'confusion_matrix': None,
        'model_path': None
    }
    
    wandb.finish()
    return y_pred

    
def train_mean_baseline(y_test, window=3,
                        dataset_name='vasp',
                        wandb_run_name="Mean_baseline_run"):
    """
    Mean baseline: predicts next power class using the
    MEAN of past power values in y_test.
    """
    print("Training Mean baseline...")
    wandb.init(project="runtime_power_pred", name=wandb_run_name, reinit=True)

    print("y_test shape before:", np.asarray(y_test).shape)
    print("y_test shape after :", np.asarray(y_test).ravel().shape)

    y_test = np.asarray(y_test).astype(float).ravel()

    # actual usage
    y_true_cont = y_test[window:].reshape(-1, 1)
    y_true = bin_power(y_true_cont)

    # mean over past window
    y_pred_cont = np.array([
        y_test[i:i+window].mean()
        for i in range(len(y_test) - window)
    ]).reshape(-1, 1)

    y_pred = bin_power(y_pred_cont)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    print("\n***** Mean Baseline Results *****")
    print(f"Accuracy: {acc:.4f}")
    print(cm)
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1, 2, 3],
        zero_division=0
    ))

    wandb.log({
        "mean_baseline_accuracy": acc,
        "mean_baseline_confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=["0", "1", "2", "3"]
        )
    })

    model_dict = {
        'model': None,
        'X_test': None,
        'y_test': y_test,
        'y_train_cls':y_true,
        'y_pred': y_pred,
        'confusion_matrix': None,
        'model_path': None
    }

    wandb.finish()
    return y_pred



def train_xgb_classifier(X_train, y_train, X_test, y_test, dcgm_set1_metrics, 
                         model_type="XGB", dataset_name='vasp', wandb_run_name="XGB_classifier_run"):
    """
    XGBoost: predicts power class based with XGBoost classifier from scikit-learn library.
    """
    y_train_cls = bin_power(y_train)
    y_test_cls = bin_power(y_test)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_flat = np.nan_to_num(X_train_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_flat = np.nan_to_num(X_test_flat, nan=0.0, posinf=0.0, neginf=0.0)

    valid_train_idx = ~np.isnan(y_train_cls)
    valid_test_idx = ~np.isnan(y_test_cls)
    
    X_train_flat = X_train_flat[valid_train_idx]
    y_train_cls = y_train_cls[valid_train_idx]
    
    X_test_flat = X_test_flat[valid_test_idx]
    y_test_cls = y_test_cls[valid_test_idx]

    print("Training XGBoost classifier...")
    # print(f"Cleaned input data shape: X_train = {X_train_flat.shape}, y = {y_train_cls.shape}")

    wandb.init(project="runtime_power_pred", name=wandb_run_name, reinit=True)

    config_params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "use_label_encoder": False,
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0,
        "n_estimators": 2000,
        "learning_rate": 0.2,
        "max_depth": 9
    }

    wandb.config.update(config_params)
    xgb_clf = XGBClassifier(**config_params)

    xgb_clf.fit(
        X_train_flat, y_train_cls.ravel(),
        eval_set=[(X_train_flat, y_train_cls), (X_test_flat, y_test_cls)],
        eval_metric='mlogloss',
        early_stopping_rounds=100,
        callbacks=[WandbCallback()]
    )

    y_pred_xgb = xgb_clf.predict(X_test_flat)

    acc = accuracy_score(y_test_cls, y_pred_xgb)
    cm = confusion_matrix(y_test_cls, y_pred_xgb)
    print("\n***** XGBoost Classifier Results *****")
    print("Accuracy:", acc)
    print(cm)
    print(classification_report(y_test_cls, y_pred_xgb))

    # --- Log to wandb ---
    wandb.log({
        "accuracy": acc,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test_cls,
            preds=y_pred_xgb,
            class_names=["0", "1", "2", "3"]
        )
    })


    importances = xgb_clf.feature_importances_

    # build flattened feature names to match X_train_flat ordering (time-major then feature)
    n_timesteps = 3 #window size is 3
    dcgm_feats = dcgm_set1_metrics + ['nersc_ldms_dcgm_power_usage'] + ['User', 'JobName', 'Account', 'Category', 'req_node','req_time']
    
    
    feature_names = []
    for t in range(n_timesteps):
        for feat in dcgm_feats:
            feature_names.append(f"{feat}_t{t}")  
            
    print(f"Total feature names: {len(feature_names)}")

    score = xgb_clf.get_booster().get_score(importance_type='gain') 
    
    # build an importance array aligned with feature_names (default 0 for missing features)
    n_feats = len(feature_names)
    imp_array = np.zeros(n_feats, dtype=float)
    
    for k, v in score.items():  # keys like 'f12'
        try:
            idx = int(k[1:])
        except ValueError:
            continue
        if 0 <= idx < n_feats:
            imp_array[idx] = float(v)
            
    # normalize importances to sum to 1 (if total > 0)
    total = imp_array.sum()
    if total > 0:
        norm_imp = imp_array / total
    else:
        norm_imp = imp_array  # all zeros
        
    # map back to feature names and sort
    mapped = {feature_names[i]: norm_imp[i] for i in range(n_feats)}
    sorted_feats = sorted(mapped.items(), key=lambda x: x[1], reverse=True)


     # print top features
    print("Top features (normalized gain):")
    for feat, val in sorted_feats[:15]:
        print(f"{feat}: {val:.4f}")

    fi_df = pd.DataFrame(sorted_feats, columns=['feature', 'importance'])
    wandb.log({"feature_importance_table": wandb.Table(dataframe=fi_df)})
    wandb.finish()


    model_dict = {
        'model': xgb_clf,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_xgb,
        'confusion_matrix': cm,
        'model_path': model_path
    }
    
    return model_dict


def train_lgb_classifier(X_train, y_train, X_test, y_test, dcgm_set1_metrics,
                         model_type="LGB", dataset_name='vasp', wandb_run_name="LGB_classifier_run"):
    """
    LightGBM: predicts power class based with LightGBM classifier from scikit-learn library.
    """
    y_train_cls = bin_power(y_train)
    y_test_cls = bin_power(y_test)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_flat = np.nan_to_num(X_train_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_flat = np.nan_to_num(X_test_flat, nan=0.0, posinf=0.0, neginf=0.0)

    valid_train_idx = ~np.isnan(y_train_cls)
    valid_test_idx = ~np.isnan(y_test_cls)
    
    X_train_flat = X_train_flat[valid_train_idx]
    y_train_cls = y_train_cls[valid_train_idx]
    
    X_test_flat = X_test_flat[valid_test_idx]
    y_test_cls = y_test_cls[valid_test_idx]

    print("Training LightGBM classifier...")
    # print(f"Cleaned input data shape: X_train = {X_train_flat.shape}, y = {y_train_cls.shape}")

    wandb.init(project="runtime_power_pred", name=wandb_run_name, reinit=True)

    config_params = {
        "objective": "multiclass",
        "num_class": 4,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "max_depth": 9,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "metric": "multi_logloss"
    }

    wandb.config.update(config_params)
    lgb_clf = lgb.LGBMClassifier(**config_params)

    # Train with early stopping
    lgb_clf.fit(
        X_train_flat, y_train_cls.ravel(),
        eval_set=[(X_train_flat, y_train_cls), (X_test_flat, y_test_cls)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    y_pred_lgb = lgb_clf.predict(X_test_flat)

    acc = accuracy_score(y_test_cls, y_pred_lgb)
    cm = confusion_matrix(y_test_cls, y_pred_lgb)
    print("\n***** LightGBM Classifier Results *****")
    print("Accuracy:", acc)
    print(cm)
    print(classification_report(y_test_cls, y_pred_lgb))

    wandb.log({
        "accuracy": acc,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test_cls,
            preds=y_pred_lgb,
            class_names=["0", "1", "2", "3"]
        )
    })

    # window size
    n_timesteps = 3
    dcgm_feats = dcgm_set1_metrics + ['nersc_ldms_dcgm_power_usage'] + ['User', 'JobName', 'Account', 'Category', 'req_node','req_time']
    
    feature_names = []
    for t in range(n_timesteps):
        for feat in dcgm_feats:
            feature_names.append(f"{feat}_t{t}")
    
    print(f"Total feature names: {len(feature_names)}")
    
    imp_array = lgb_clf.booster_.feature_importance(importance_type='gain')
    
    # sanity check
    assert len(imp_array) == len(feature_names), \
        f"Mismatch: model has {len(imp_array)} features, names has {len(feature_names)}"
    
    # normalize
    total = imp_array.sum()
    norm_imp = imp_array / total if total > 0 else imp_array
    
    # map & sort
    mapped = dict(zip(feature_names, norm_imp))
    sorted_feats = sorted(mapped.items(), key=lambda x: x[1], reverse=True)
    
    print("Top features (normalized gain):")
    for feat, val in sorted_feats[:15]:
        print(f"{feat}: {val:.4f}")
    
    fi_df = pd.DataFrame(sorted_feats, columns=['feature', 'importance'])
    wandb.log({"feature_importance_table": wandb.Table(dataframe=fi_df)})
    wandb.finish()


    model_dict = {
        'model': lgb_clf,
        'X_test': X_test,
        'y_test': y_test,
        'y_train_cls':y_train_cls,
        'y_pred': y_pred_lgb,
        'confusion_matrix': cm,
        'model_path': model_path
    }
    
    return model_dict
    
def train_lstm_classifier(X_train, y_train, X_test, y_test, 
                          wandb_run_name="LSTM_classifier_run"):
    """
    LSTM: predicts power class based with a simple LSTM model.
    """
    y_train_cls = bin_power(y_train)
    y_test_cls = bin_power(y_test)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    valid_train_idx = ~np.isnan(y_train_cls)
    valid_test_idx = ~np.isnan(y_test_cls)

    X_train = X_train[valid_train_idx]
    y_train_cls = y_train_cls[valid_train_idx]
    X_test = X_test[valid_test_idx]
    y_test_cls = y_test_cls[valid_test_idx]

    print("Training LSTM classifier...")
    print(f"Cleaned input data shape: X_train = {X_train.shape}, y = {y_train_cls.shape}")
    
    wandb.init(project="runtime_power_pred", name=wandb_run_name, reinit=True)

    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    n_classes = 4
    lstm_units = 64
    dropout = 0.2
    batch_size = 32
    epochs = 100
    early_stopping_patience = 10

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(n_timesteps, n_features)))
    model.add(LSTM(lstm_units, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=early_stopping_patience, 
        restore_best_weights=True
    )
    
    wandb_callback = KerasWandbCallback(
        log_batch_frequency=None, 
        log_model=False  
    )
    model.fit(
        X_train, y_train_cls,
        validation_data=(X_test, y_test_cls),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, KerasWandbCallback()],
        verbose=1
    )

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test_cls, y_pred)
    cm = confusion_matrix(y_test_cls, y_pred)
    print("\n***** LSTM Classifier Results *****")
    print("Accuracy:", acc)
    print(cm)
    print(classification_report(y_test_cls, y_pred))

    wandb.log({
        "accuracy": acc,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test_cls,
            preds=y_pred,
            class_names=["0", "1", "2", "3"]
        )
    })

    
    wandb.finish()

    return {
        "model": model,
        "y_pred": y_pred,
        "y_test": y_test_cls,
        "confusion_matrix": cm,
        "model_path": model_path
    }


if __name__ == "__main__":
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Script for model training")
    parser.add_argument("CodeGroup", help="Options are: vasp, lammps, chroma, e3sm, espresso, atlas.")
    parser.add_argument("Model", help="Time series prediction model (XGB, LGB, LSTM, Max, Mean)")
    parser.add_argument("RunName", help="Name of the experiment")
    parser.add_argument("--num_jobs", type=int, help="Number of jobs to train (optional)")
    args = parser.parse_args()

    # Load and prepare data -- time-series DCGM metrics
    print("Loading DCGM data...")
    if args.CodeGroup == 'vasp':
        chunks = [pd.read_parquet(f"../ml_training_files/VASP_dcgm_data_part_{i+1}.parquet") 
                  for i in range(10)]
        app_job_df = pd.concat(chunks, ignore_index=True)
    elif args.CodeGroup == 'chroma':
        chunks = [pd.read_parquet(f"../ml_training_files/CHROMA_dcgm_data_part_{i+1}.parquet") 
                  for i in range(10)]
        app_job_df = pd.concat(chunks, ignore_index=True)
    elif args.CodeGroup == 'lammps':
        app_job_df = pd.read_parquet(f"../ml_training_files/LAMMPS_data.parquet")
    elif args.CodeGroup == 'espresso':
        app_job_df = pd.read_parquet(f"../ml_training_files/ESPRESSO_data.parquet")
    elif args.CodeGroup == 'atlas':
        app_job_df = pd.read_parquet(f"../ml_training_files/ATLAS_data.parquet")
    elif args.CodeGroup == 'e3sm':
        app_job_df = pd.read_parquet(f"../ml_training_files/E3SM_data.parquet")
    else:
        raise ValueError(f"Unknown dataset name: {args.CodeGroup}")
        
    
    # Filter valid data
    app_job_df["MetricsCheck"] = app_job_df.apply(check_metrics, axis=1)
    ok_df = app_job_df[app_job_df["MetricsCheck"].apply(lambda x: x.get("status") == "ok")].copy()
    print(f"Combined DataFrame shape: {ok_df.shape}")

    # Take last n jobs if provided in CLI arguments -- Not used in the experiments for the manuscript.
    exp_df = ok_df.iloc[-args.num_jobs:] if args.num_jobs else ok_df.copy()
    cleaned_df = exp_df.apply(clean_and_compare, axis=1)

    dcgm_set1_metrics = ['dram_active', 'fb_free', 'fb_used',
                         'fp64_active', 'tensor_active', 
                         'gpu_utilization', 'mem_util',
                         'sm_active', 'sm_occupancy']
    slurm_features = ['User', 'Category', 'JobName', 'Account', 'req_node', 'req_time']
    # print(f"Using DCGM Set1 metrics: {dcgm_set1_metrics} ({len(dcgm_set1_metrics)} features)")


    train_df, test_df = train_test_split(cleaned_df, test_size=0.2, 
                                        random_state=42, shuffle=False)

    train_df = train_df.copy()
    test_df = test_df.copy()

    cat_cols = ['User', 'Category', 'JobName', 'Account']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])
    
    # print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    print("Preparing sequences...")
    X_train_lstm, X_train_xgb, y_train = prepare_multivariate_sequences(
        train_df, dcgm_set1_metrics,
        target_metric='nersc_ldms_dcgm_power_usage',
        window=3
    )
    
    X_test_lstm, X_test_xgb, y_test = prepare_multivariate_sequences(
        test_df, dcgm_set1_metrics, 
        target_metric='nersc_ldms_dcgm_power_usage',
        window=3
    )
    
    print(f"\nTraining {args.Model} model...")
    
    if args.Model == 'XGB':
        model_dict = train_xgb_classifier(X_train_xgb, y_train, X_test_xgb, y_test, 
                                         dcgm_set1_metrics, "XGB", args.CodeGroup , args.RunName)
    elif args.Model == 'LGB':
        model_dict = train_lgb_classifier(X_train_xgb, y_train, X_test_xgb, y_test, 
                                          dcgm_set1_metrics, "LGB", args.CodeGroup, wandb_run_name=args.RunName)    
    elif args.Model == 'LSTM':
        model_dict = train_lstm_classifier(X_train_lstm, y_train, X_test_lstm, y_test, 
                                          wandb_run_name=args.RunName)
    elif args.Model == 'Max':
        model_dict = train_max_baseline(y_test, 
                                       wandb_run_name=args.RunName)
    elif args.Model == 'Mean':
        model_dict = train_mean_baseline(y_test, 
                                        wandb_run_name=args.RunName)
    else:
        raise ValueError(f"Unknown model type: {args.Model}")

    print(f"\n Training complete. Results saved.")
    



        
    
    