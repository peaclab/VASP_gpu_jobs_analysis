import numpy as np
import pandas as pd


def prepare_multivariate_sequences(metrics_df, dcgm_input_features,
                                   target_metric='nersc_ldms_dcgm_power_usage',
                                   window=3,  # number of timesteps per sequence
                                   n_gpus=4):
    X_lstm_list = []
    X_xgb_list = []
    y_list = []

    for _, row in metrics_df.iterrows():
        s1 = row.get('Set1Metrics')
        s2 = row.get('Set2Metrics')
        
        arrs = []
        lengths = []
        for f in dcgm_input_features:
            feat = f'nersc_ldms_dcgm_{f}'
            arr = np.asarray(s1.get(feat, []), dtype=float)
            if arr.size == 0: 
                continue
            arrs.append(arr)
            lengths.append(len(arr))
        
        arr = np.asarray(s2.get('nersc_ldms_dcgm_power_usage', []), dtype=float)
        if arr.size == 0: 
            continue
        arrs.append(arr)
        lengths.append(len(arr))    
            
        target_arr = np.asarray(s2.get(target_metric, []), dtype=float)
        lengths.append(len(target_arr))
        n_timesteps_raw = min(lengths)

        hostnames = s2.get('hostname', [])
        if isinstance(hostnames, list):
            n_nodes = len(set(hostnames))
        else:
            n_nodes = 1

        total_gpus = n_nodes * n_gpus

        if n_timesteps_raw % total_gpus != 0:
            n_timesteps_raw = (n_timesteps_raw // total_gpus) * total_gpus
        
        n_timesteps = n_timesteps_raw // total_gpus
        usable = n_timesteps - window
        if usable <= 0:
            continue

        # Reshape and average across GPUs
        arrs_reshaped = []
        for a in arrs:
            a_trimmed = a[:n_timesteps_raw]
            a_2d = a_trimmed.reshape(n_timesteps, total_gpus)
            a_avg = a_2d.mean(axis=1)
            arrs_reshaped.append(a_avg)
        
        target_trimmed = target_arr[:n_timesteps_raw]
        target_2d = target_trimmed.reshape(n_timesteps, total_gpus)
        target_avg = target_2d.mean(axis=1)

        stacked = np.vstack(arrs_reshaped).T  # (n_timesteps, n_features)

        # static features
        static_vals = []
        for s_feat in ['User', 'JobName', 'Account', 'Category', 'req_node','req_time']:
            val = row.get(s_feat)
            static_vals.append(float(val) if val is not None else 0.0)
        static_vals = np.array(static_vals, dtype=float)

        # --- build sequences
        for i in range(usable):
            seq_window = stacked[i:i+window]  # (window, n_features)

            # LSTM input: keeping 3D
            static_repeated = static_vals.reshape(1, -1).repeat(window, axis=0)
            seq_lstm = np.concatenate([seq_window, static_repeated], axis=1)  # (window, n_features + static)
            X_lstm_list.append(seq_lstm)

            # XGBoost input: flattening sequence + static
            seq_flat = seq_lstm.flatten()
            X_xgb_list.append(seq_flat)

            # target variable
            y_list.append(target_avg[i+window])

    X_lstm = np.stack(X_lstm_list)  # (samples, window, n_features + static)
    X_xgb = np.stack(X_xgb_list)    # (samples, window*(n_features + static))
    y = np.array(y_list).reshape(-1, 1)

    # print(f"LSTM input shape: {X_lstm.shape}")
    # print(f"XGBoost input shape: {X_xgb.shape}")
    # print(f"Target shape: {y.shape}")

    return X_lstm, X_xgb, y


    