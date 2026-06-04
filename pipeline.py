import os
import re

# Suppress TensorFlow startup informational warnings to keep console output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Mathematical libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicHermiteSpline
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, precision_recall_curve, auc)

# Modeling Architecture Libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, MultiHeadAttention, LayerNormalization, 
                                     Dropout, GlobalAveragePooling1D, LSTM, RepeatVector, TimeDistributed, Embedding)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Configure Visual EDA Theme Styling
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = [16, 12]

# =====================================================================
# 2. DATA CLEANING & STANDARDIZATION
# =====================================================================
def convert_dms_to_decimal(cell_value):
    if pd.isna(cell_value) or str(cell_value).strip() == "" or str(cell_value).strip().lower() == "nan":
        return np.nan
    coord_str = str(cell_value).strip().replace('"', '')  
    if not any(char in coord_str for char in ['deg', "'", '"', 'N', 'S', 'E', 'W', 'n', 's', 'e', 'w']):
        try: return float(coord_str)
        except ValueError: return np.nan
    numbers = [float(n) for n in re.findall(r'[-+]?\d*\.\d+|\d+', coord_str)]
    direction = re.findall(r'[NSEWnsew]', coord_str)
    if not numbers: return np.nan
    degrees = numbers[0] if len(numbers) >= 1 else 0.0
    minutes = numbers[1] if len(numbers) >= 2 else 0.0
    seconds = numbers[2] if len(numbers) >= 3 else 0.0
    decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if direction and direction[0].upper() in ['S', 'W']:
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def normalize_accelerometer(cell_value):
    if pd.isna(cell_value) or str(cell_value).strip() == "" or str(cell_value).strip().lower() == "nan":
        return np.nan
    val_str = str(cell_value).strip().replace('"', '')
    numerical_parts = re.findall(r'[-+]?\d*\.\d+|\d+', val_str)
    if len(numerical_parts) >= 3:
        try: return np.sqrt(float(numerical_parts[0])**2 + float(numerical_parts[1])**2 + float(numerical_parts[2])**2)
        except ValueError: return 0.0
    elif len(numerical_parts) == 1:
        try: return abs(float(numerical_parts[0]))
        except ValueError: return 0.0
    return 0.0

class TrajectoryKalmanFilter:
    def __init__(self, dt=1.0):
        self.x = np.zeros((4, 1))
        self.F = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self.P = np.eye(4) * 1.0
        self.R = np.eye(2) * 0.0001  
        self.Q = np.eye(4) * 0.001   

    def initialize_state(self, init_lat, init_lon):
        self.x = np.array([[init_lat], [init_lon], [0.0], [0.0]])

    def process_step(self, measured_lat, measured_lon):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        z = np.array([[measured_lat], [measured_lon]])
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
        return self.x[0, 0], self.x[1, 0]

def build_transformer_sequences(df_src, X_arr, scaled_targets, mask_arr, seq_len):
    X_seq, y_seq = [], []
    start_idx = 0
    for _, group in df_src.groupby('SourceFile'):
        g_len = len(group)
        if g_len <= seq_len:
            start_idx += g_len
            continue
        g_x = X_arr[start_idx : start_idx + g_len]
        g_mask = mask_arr[start_idx : start_idx + g_len]
        g_targets = scaled_targets[start_idx : start_idx + g_len]
        
        for i in range(g_len - seq_len):
            X_seq.append(g_x[i : i + seq_len])
            target_vals = g_targets[i + seq_len]
            target_mask = g_mask[i + seq_len]
            combined_y = np.array([target_vals[0], target_vals[1], target_vals[2], target_mask])
            y_seq.append(combined_y)
        start_idx += g_len
    return np.array(X_seq), np.array(y_seq)

def weighted_forensic_mse(y_true, y_pred):
        loss_weights = tf.constant([1.0, 1.0, 5.0], dtype=tf.float32)
        squared_errors = tf.square(y_true - y_pred)
        weighted_errors = squared_errors * loss_weights
        return tf.reduce_mean(weighted_errors)

# =====================================================================
# MAIN RUNTIME PIPELINE EXECUTION ENGINE
# =====================================================================
def run_forensic_pipeline(combined_data_csv, audit_log_csv, output_features_csv, plot_export_dir="."):
    if not os.path.exists(plot_export_dir):
        os.makedirs(plot_export_dir)

    df_raw = pd.read_csv(combined_data_csv)
    print("\n[*] Initializing Data Cleaning & Standardization...")
    
    df_raw.columns = [col.strip().replace('"', '') for col in df_raw.columns]
    df_raw['GPSLatitude_Clean'] = df_raw['GPSLatitude'].apply(convert_dms_to_decimal)
    df_raw['GPSLongitude_Clean'] = df_raw['GPSLongitude'].apply(convert_dms_to_decimal)
    df_raw['Accelerometer_Normalized'] = df_raw['Accelerometer'].apply(normalize_accelerometer)
    df_raw['GPSSpeed'] = pd.to_numeric(df_raw['GPSSpeed'].astype(str).str.replace('"', '').str.strip(), errors='coerce')
    df_raw['Speed_m_s'] = df_raw['GPSSpeed'] / 3.6

    df_raw['Timestamp_Unified'] = df_raw['GPSDateTime'].apply(
        lambda x: str(x).replace('Z', '').replace(':', '-', 2).replace('"', '').strip() if pd.notna(x) else x
    )
    df_raw['Timestamp_Unified'] = pd.to_datetime(df_raw['Timestamp_Unified'], errors='coerce')
    df_raw['GPS_Date'] = df_raw['Timestamp_Unified'].dt.date
    df_raw['GPS_Time'] = df_raw['Timestamp_Unified'].dt.time
    df_raw = df_raw.sort_values(by=["SourceFile", "Timestamp_Unified"]).reset_index(drop=True)

    # First null vallue check
    print("\n--- NULL VALUE CHECK ---")
    print(df_raw.isna().sum())

    # =====================================================================
    # FEATURE ENGINEERING 
    # =====================================================================
    print("\n[*] Initializing Feature Engineering...")
    feature_blocks = []

    for video_id, group in df_raw.groupby("SourceFile"):
        group = group.copy()
        group['Time_Delta_Cumulative'] = (group['Timestamp_Unified'] - group['Timestamp_Unified'].iloc[0]).dt.total_seconds().fillna(0.0)
        dt = group['Timestamp_Unified'].diff().dt.total_seconds().fillna(1.0).replace(0, 1.0)
        
        # BERT4Traj: We predict Deltas, not raw Absolute locations
        group['Delta_Lat'] = group['GPSLatitude_Clean'].diff().fillna(0.0)
        group['Delta_Lon'] = group['GPSLongitude_Clean'].diff().fillna(0.0)
        
        group['Bearing_Rad'] = np.arctan2(group['Delta_Lon'], group['Delta_Lat']).fillna(0.0)
        temp_speed = group['Speed_m_s'].ffill().bfill().fillna(0.0)
        group['Velocity x'] = temp_speed * np.cos(group['Bearing_Rad'])
        group['velocity y'] = temp_speed * np.sin(group['Bearing_Rad'])
        
        group['Calculated_Acceleration'] = temp_speed.diff().fillna(0.0) / dt
        group['Turn_Rate_Rad_s'] = np.unwrap(group['Bearing_Rad'])
        group['Turn_Rate_Rad_s'] = group['Turn_Rate_Rad_s'].diff().fillna(0.0) / dt
        group['Calculated_Jerk'] = group['Calculated_Acceleration'].diff().fillna(0.0) / dt
        
        group['GPS_Missing_Flag'] = np.where(group['GPSLatitude_Clean'].isna(), 1.0, 0.0)
        group['Speed_Missing_Flag'] = np.where(group['Speed_m_s'].isna(), 1.0, 0.0)
        group['Accelerometer_Normalized'] = group['Accelerometer_Normalized'].ffill().bfill().fillna(1.0)
        group['Acc_Rolling_Variance'] = group['Accelerometer_Normalized'].rolling(window=3, min_periods=1).var().fillna(0.0)
        
        feature_blocks.append(group)

    df_export = pd.concat(feature_blocks, ignore_index=True).replace([np.inf, -np.inf], 0.0)
    df_export.loc[df_export['GPS_Missing_Flag'] == 1.0, ['GPSLatitude_Clean', 'GPSLongitude_Clean']] = np.nan
    df_export.loc[df_export['Speed_Missing_Flag'] == 1.0, 'Speed_m_s'] = np.nan


    column_schema = [
        'SourceFile', 'GPS_Date', 'GPS_Time', 'Timestamp_Unified', 'Time_Delta_Cumulative',
        'GPSLatitude_Clean', 'GPSLongitude_Clean', 'Delta_Lat', 'Delta_Lon', 'Bearing_Rad', 'Speed_m_s', 
        'Velocity x', 'velocity y', 'Calculated_Acceleration', 'Turn_Rate_Rad_s', 
        'Calculated_Jerk', 'Accelerometer_Normalized', 'Acc_Rolling_Variance',
        'GPS_Missing_Flag', 'Speed_Missing_Flag'
    ]
    df_export = df_export[column_schema].copy()

    #check null vallue after feature engineering
    print("\n--- FINAL CHECKING NULL VALUE AFTER FEATURE ENGINEERING---")
    print(df_export.isna().sum())

    df_export.to_csv(output_features_csv, index=False)

    # =====================================================================
    # 4. EXPLORATORY DATA ANALYSIS (EDA)
    # =====================================================================
    print("\n" + "="*70)
    print("=== EXPLORATORY DATA ANALYSIS (EDA) ===")
    print("="*70)
    print(f"Total Rows in Entire Pipeline Dataset: {len(df_export)}")
    print(f"Total Tracked Unique Video Clips: {df_export['SourceFile'].nunique()}")
    
    print("\nStatistical Distribution:")
    print(df_export.describe().T)
    
    # Select the video segment containing the most data entries for isolated visualization
    target_sample_video = df_export['SourceFile'].value_counts().index[0]
    df_plot_sample = df_export[df_export['SourceFile'] == target_sample_video].sort_values('Time_Delta_Cumulative')

    print(f"\n[VISUAL MATCH FOUND]: Generating Simple EDA for Isolate Clip:")
    print(f"   File Target: {os.path.basename(target_sample_video)}")
    print(f"   Target Date Context: {df_plot_sample['GPS_Date'].iloc[0]}")

    # CHART 1: Speed Profile Curve Graph
    plt.figure(figsize=(10, 4))
    plt.plot(df_plot_sample['Time_Delta_Cumulative'], df_plot_sample['Speed_m_s'], 
             color='darkorange', linewidth=3, label='Vehicle Speed')
    plt.title(f"Vehicle Speed Profile Over Time [{os.path.basename(target_sample_video)}]", fontsize=13, fontweight='bold')
    plt.xlabel('Time Progressions (Seconds)', fontsize=11)
    plt.ylabel('Speed (m/s)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_export_dir, 'eda_speed_profile.png'))
    plt.close()

    # CHART 2: Overhead GPS Track Map Route
    plt.figure(figsize=(7, 7))
    plt.plot(df_plot_sample['GPSLongitude_Clean'], df_plot_sample['GPSLatitude_Clean'], 
             color='royalblue', linewidth=3, label='Driven Path')
    plt.scatter(df_plot_sample['GPSLongitude_Clean'].iloc[0], df_plot_sample['GPSLatitude_Clean'].iloc[0], 
                color='green', s=160, zorder=5, label='ROUTE START')
    plt.scatter(df_plot_sample['GPSLongitude_Clean'].iloc[-1], df_plot_sample['GPSLatitude_Clean'].iloc[-1], 
                color='crimson', s=160, zorder=5, label='ROUTE END')
    plt.title(f"Vehicle Driving Route Map (GPS Tracking Coordinates)", fontsize=13, fontweight='bold')
    plt.xlabel('Longitude', fontsize=11)
    plt.ylabel('Latitude', fontsize=11)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_export_dir, 'eda_gps_track.png'))
    plt.close()

    # CHART 3: Acceleration & Braking Profile (Force Analysis)
    plt.figure(figsize=(10, 4))
    plt.fill_between(df_plot_sample['Time_Delta_Cumulative'], df_plot_sample['Calculated_Acceleration'], 0,
                     where=(df_plot_sample['Calculated_Acceleration'] >= 0), color='green', alpha=0.3, label='Acceleration')
    plt.fill_between(df_plot_sample['Time_Delta_Cumulative'], df_plot_sample['Calculated_Acceleration'], 0,
                     where=(df_plot_sample['Calculated_Acceleration'] < 0), color='crimson', alpha=0.3, label='Braking / Deceleration')
    plt.plot(df_plot_sample['Time_Delta_Cumulative'], df_plot_sample['Calculated_Acceleration'], color='black', linewidth=1.5, alpha=0.6)
    plt.axhline(0, color='gray', linestyle='-', linewidth=1)
    plt.title(f"Vehicle Acceleration & Braking Forces", fontsize=13, fontweight='bold')
    plt.xlabel('Time Elapsed (Seconds)', fontsize=11)
    plt.ylabel('Acceleration (m/s²)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_export_dir, 'eda_acceleration_profile.png'))
    plt.close()

    # CHART 4: Turn Rate Characteristics (Steering Behavior)
    plt.figure(figsize=(10, 4))
    plt.plot(df_plot_sample['Time_Delta_Cumulative'], df_plot_sample['Turn_Rate_Rad_s'], 
             color='purple', linewidth=2.5, label='Turn Rate')
    plt.axhline(0, color='gray', linestyle='-', linewidth=1)
    plt.title(f"Vehicle Turn Rate & Steering Tracking", fontsize=13, fontweight='bold')
    plt.xlabel('Time Elapsed (Seconds)', fontsize=11)
    plt.ylabel('Turn Rate (Radians/Sec)', fontsize=11)
    plt.text(df_plot_sample['Time_Delta_Cumulative'].min() + 2, 0.05, '🔄 Turning Left', color='purple', fontsize=10, fontweight='bold')
    plt.text(df_plot_sample['Time_Delta_Cumulative'].min() + 2, -0.05, 'Turning Right 🔄', color='purple', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_export_dir, 'eda_turn_rate.png'))
    plt.close()

    print("=== DATA PREPROCESSING AND EDA COMPLETELY FINISHED ===")

    # =====================================================================
    # INTERPOLATION AND TRIANGULATION MODELLING FOR INCIDENT RECOSNTRUCTION
    # =====================================================================
    print("\n" + "="*70)
    print("=== MODEL CONFIGURATION & TRAINING ===")
    print("="*70)

    # =====================================================================
    # BERT4Traj STYLE TRANSFORMER MODELLING 
    # =====================================================================
    
    lstm_predictors = [
        'Time_Delta_Cumulative', 'Bearing_Rad', 'Velocity x', 'velocity y', 
        'Calculated_Acceleration', 'Turn_Rate_Rad_s', 'Calculated_Jerk', 
        'Accelerometer_Normalized', 'Acc_Rolling_Variance', 'GPS_Missing_Flag', 'Speed_Missing_Flag'
    ]
    
    # BERT4Traj TARGETS: Deltas instead of Absolute Location
    lstm_targets = ['Delta_Lat', 'Delta_Lon', 'Speed_m_s']
    
    y_lstm_raw_vals = df_export[['GPSLatitude_Clean', 'GPSLongitude_Clean', 'Speed_m_s']].values
    valid_mask_raw = (np.isnan(y_lstm_raw_vals[:, 0]) == False) & (np.isnan(y_lstm_raw_vals[:, 2]) == False)
    valid_mask_raw = valid_mask_raw.astype(np.float32)

    df_lstm_y_filled = df_export[lstm_targets].ffill().bfill().fillna(0.0).values

    # USCALERS
    scaler_X = StandardScaler() # Standard scaling preserves negative/positive direction
    scaler_y = StandardScaler() 
    scaler_ae = MinMaxScaler()  # Autoencoder keeps MinMax

    X_lstm_scaled = scaler_X.fit_transform(df_export[lstm_predictors].values)
    y_lstm_scaled = scaler_y.fit_transform(df_lstm_y_filled)

    SEQUENCE_LENGTH = 50

    X_trans_m, y_trans_m = build_transformer_sequences(
        df_export, X_lstm_scaled, y_lstm_scaled, valid_mask_raw, SEQUENCE_LENGTH
    )

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_trans_m, y_trans_m, test_size=0.20, random_state=42)

    print("\n[*] Initializing BERT4Traj Architecture...")
    feature_input = Input(shape=(SEQUENCE_LENGTH, len(lstm_predictors)), name="telemetry_features")

    # First Layer Normalization and Multi-Head Attention
    norm_1 = LayerNormalization(epsilon=1e-6)(feature_input)
    # Increase dropout inside the attention mechanism to prevent path memorization
    attention_out, attention_weights = MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.3  # <--- Increased from 0.1 to 0.3
    )(norm_1, norm_1, return_attention_scores=True)
    
    attention_residual = tf.keras.layers.Add()([feature_input, attention_out])
    
    # Second Layer Normalization and Feed-Forward Network
    norm_2 = LayerNormalization(epsilon=1e-6)(attention_residual)
    
    # Inject L2 regularization into the Dense layers to penalize massive weight spikes
    dense_ff = Dense(
        64, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4) # <--- Added L2 Regularization
    )(norm_2)
    
    # Increase feed-forward dropout
    dropout_ff = Dropout(0.3)(dense_ff)  # <--- Increased from 0.1 to 0.3
    
    ff_out = Dense(len(lstm_predictors))(dropout_ff)
    transformer_block_out = tf.keras.layers.Add()([attention_residual, ff_out])
    
    # Global Pooling and Output Head
    global_pooling = GlobalAveragePooling1D()(transformer_block_out)
    
    dense_final = Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4) # <--- Added L2 Regularization
                        )(global_pooling)
    
    coordinate_output = Dense(3, name="Delta_Trajectory_Outputs")(dense_final)
    positions = tf.range(start=0, limit=SEQUENCE_LENGTH, delta=1)
    position_embedding = Embedding(input_dim=SEQUENCE_LENGTH, output_dim=len(lstm_predictors))(positions)
    
    x = feature_input + position_embedding
    attention_block = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    norm_x = LayerNormalization(epsilon=1e-6)(attention_block + feature_input)
    
    ffn = Dense(128, activation="relu")(norm_x)
    ffn = Dense(len(lstm_predictors))(ffn)
    norm_x2 = LayerNormalization(epsilon=1e-6)(ffn + norm_x)
    
    pooling = GlobalAveragePooling1D()(norm_x2)
    dense_proj = Dense(64, activation='relu')(pooling)
    dense_proj = Dense(32, activation='relu')(dense_proj)
    coordinate_output = Dense(len(lstm_targets), name="coordinate_output")(dense_proj)

    transformer_model = Model(inputs=feature_input, outputs=coordinate_output)
    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=weighted_forensic_mse,  metrics=['mae']
    )

    # This automatically prevents overfitting. If validation loss stops dropping, it kills training.
    early_stopping_transformer = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,                 # Stop if validation loss doesn't improve for 3 epochs
        restore_best_weights=True   # Automatically roll back to the absolute lowest validation loss epoch
    )

    print("[*] Training Transformer Parameters...")
    t_history = transformer_model.fit(
        X_train_t, 
        y_train_t[:, :3], 
        validation_data=(X_test_t, y_test_t[:, :3]), 
        epochs=15,                 
        batch_size=64, 
        callbacks=[early_stopping_transformer], 
        verbose=1
    )

    # =====================================================================
    # KALMAN FILTER ENGINE & CUBIC HERMITE SMOOTHING
    # =====================================================================
    print("\n[*] Initializing Kalman Filtering & Hermite Splines...")

    trans_predictions_scaled = transformer_model.predict(X_trans_m, verbose=0)
    pred_unscaled_matrix = scaler_y.inverse_transform(trans_predictions_scaled)

    # GPS ANCHORING TO PREVENT DRIFT ---
    start_idx = SEQUENCE_LENGTH - 1
    true_lats = df_export['GPSLatitude_Clean'].ffill().bfill().values[start_idx:]
    true_lons = df_export['GPSLongitude_Clean'].ffill().bfill().values[start_idx:]
    gps_missing = df_export['GPS_Missing_Flag'].values[start_idx:]

    reconstructed_lat = []
    reconstructed_lon = []
    
    curr_lat = true_lats[0]
    curr_lon = true_lons[0]

    # Trust the Neural Network 95% for smoothness, but pull towards True GPS 5% to destroy drift
    alpha = 0.99 

    for i in range(len(pred_unscaled_matrix)):
        predicted_step_lat = pred_unscaled_matrix[i, 0]
        predicted_step_lon = pred_unscaled_matrix[i, 1]
        
        if gps_missing[i] == 0.0:
            curr_lat = alpha * (curr_lat + predicted_step_lat) + (1.0 - alpha) * true_lats[i]
            curr_lon = alpha * (curr_lon + predicted_step_lon) + (1.0 - alpha) * true_lons[i]

        else:
            # Dead-reckon blindly only when GPS drops completely
            curr_lat += predicted_step_lat
            curr_lon += predicted_step_lon
            
        reconstructed_lat.append(curr_lat)
        reconstructed_lon.append(curr_lon)

    reconstructed_absolute_lat = np.array(reconstructed_lat)
    reconstructed_absolute_lon = np.array(reconstructed_lon)

    kf = TrajectoryKalmanFilter(dt=1.0)
    kf.initialize_state(reconstructed_absolute_lat[0], reconstructed_absolute_lon[0])
    
    kalman_sanitized = []
    for step in range(len(reconstructed_absolute_lat)):
        k_lat, k_lon = kf.process_step(reconstructed_absolute_lat[step], reconstructed_absolute_lon[step])
        kalman_sanitized.append([k_lat, k_lon])
    kalman_sanitized = np.array(kalman_sanitized)

    timesteps = np.arange(len(kalman_sanitized))
    lat_tangents = np.gradient(kalman_sanitized[:, 0])
    lon_tangents = np.gradient(kalman_sanitized[:, 1])

    hermite_spline_lat = CubicHermiteSpline(timesteps, kalman_sanitized[:, 0], lat_tangents)
    hermite_spline_lon = CubicHermiteSpline(timesteps, kalman_sanitized[:, 1], lon_tangents)

    final_polished_lat = hermite_spline_lat(timesteps)
    final_polished_lon = hermite_spline_lon(timesteps)

    # =====================================================================
    # AUTOENCODER MODELLING
    # =====================================================================
    print("\n[*] Constructing Anomaly Detection Autoencoder...")

    ae_features = ['Speed_m_s', 'Bearing_Rad', 'Velocity x', 'velocity y', 'Calculated_Acceleration', 'Turn_Rate_Rad_s', 'Calculated_Jerk', 'Accelerometer_Normalized', 'Acc_Rolling_Variance']
    lat_meters = final_polished_lat * 111000.0
    lon_meters = final_polished_lon * (111000.0 * np.cos(np.radians(np.mean(final_polished_lat))))
    dt_step = 1.0
    
    def rolling_smooth(arr, window_size=10):
        """Applies a smoothing filter to destroy noisy sensor spikes."""
        return pd.Series(arr).rolling(window=window_size, min_periods=1).mean().values

    v_x_recon = np.gradient(lon_meters, dt_step)
    v_y_recon = np.gradient(lat_meters, dt_step)
    
    raw_speed = np.sqrt(v_x_recon**2 + v_y_recon**2)
    bearing_recon = np.arctan2(v_x_recon, v_y_recon)
    
    # ROLLING AVERAGE FILTER APPLIED TO FIX FALSE POSITIVES
    speed_recon = rolling_smooth(raw_speed, window_size=5)
    turn_rate_recon = rolling_smooth(np.gradient(np.unwrap(bearing_recon), dt_step), window_size=5)
    accel_recon = rolling_smooth(np.gradient(speed_recon, dt_step), window_size=5)
    jerk_recon = rolling_smooth(np.gradient(accel_recon, dt_step), window_size=5)
    
    ae_processed_matrix = np.column_stack([
        speed_recon, bearing_recon, v_x_recon, v_y_recon,
        accel_recon, turn_rate_recon, jerk_recon, np.ones_like(speed_recon)*1.0, np.zeros_like(speed_recon)
    ])
    
    ae_scaled_reconstructed = scaler_ae.fit_transform(ae_processed_matrix)
    
    X_ae_pipeline_windows, unscaled_ae_windows = [], []
    for i in range(len(ae_scaled_reconstructed) - SEQUENCE_LENGTH):
        X_ae_pipeline_windows.append(ae_scaled_reconstructed[i : i + SEQUENCE_LENGTH])
        unscaled_ae_windows.append(ae_processed_matrix[i : i + SEQUENCE_LENGTH])
        
    X_ae_pipeline_windows = np.array(X_ae_pipeline_windows)
    unscaled_ae_windows = np.array(unscaled_ae_windows)

    ae_inputs = Input(shape=(SEQUENCE_LENGTH, len(ae_features)))
    encoder = LSTM(64, activation='tanh', return_sequences=True)(ae_inputs)
    encoder_bottleneck = LSTM(32, activation='tanh', return_sequences=False)(encoder)
    decoder = RepeatVector(SEQUENCE_LENGTH)(encoder_bottleneck)
    decoder = LSTM(32, activation='tanh', return_sequences=True)(decoder)
    ae_outputs = TimeDistributed(Dense(len(ae_features)))(decoder)

    autoencoder = Model(inputs=ae_inputs, outputs=ae_outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')

    min_len = min(len(y_trans_m), len(X_ae_pipeline_windows))
    y_test_t_adjusted = y_trans_m[:min_len]
    X_ae_pipeline_windows = X_ae_pipeline_windows[:min_len]
    unscaled_ae_windows = unscaled_ae_windows[:min_len]
    
    # Define normal physical thresholds based on percentiles, not hard limits
    accel_upper = np.percentile(unscaled_ae_windows[:, :, 4], 99)
    accel_lower = np.percentile(unscaled_ae_windows[:, :, 4], 1)
    turn_extreme = np.percentile(np.abs(unscaled_ae_windows[:, :, 5]), 99)

    gps_dropout_flag = np.any(y_test_t_adjusted[:, 3:4] == 0.0, axis=1)
    harsh_brake_flag = np.any(unscaled_ae_windows[:, :, 4] < accel_lower, axis=1)   
    aggressive_accel_flag = np.any(unscaled_ae_windows[:, :, 4] > accel_upper, axis=1) 
    sharp_turn_flag = np.any(np.abs(unscaled_ae_windows[:, :, 5]) > turn_extreme, axis=1) 
    
    true_anomalies_proxy = (gps_dropout_flag | harsh_brake_flag | aggressive_accel_flag | sharp_turn_flag).astype(int)

    normal_driving_mask = (true_anomalies_proxy == 0)
    X_train_ae_clean = X_ae_pipeline_windows[normal_driving_mask]

    if len(X_train_ae_clean) < 10: X_train_ae_clean = X_ae_pipeline_windows

    ae_tr, ae_val = train_test_split(X_train_ae_clean, test_size=0.2, random_state=42)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("[*] Training Behavioral Autoencoder...")
    ae_history = autoencoder.fit(
        ae_tr, ae_tr, epochs=15, batch_size=64, validation_data=(ae_val, ae_val), callbacks=[early_stop], verbose=1
    )

    # =====================================================================
    # PERFORMANCE METRICS MATRIX RESULTS 
    # =====================================================================
    print("\n" + "="*70)
    print("=== FINAL ACCURACY METRICS PERFORMANCE EVALUATIONS ===")
    print("="*70)

    ae_predictions = autoencoder.predict(X_ae_pipeline_windows, verbose=0)
    raw_absolute_errors = np.abs(ae_predictions - X_ae_pipeline_windows)
    raw_absolute_errors = np.nan_to_num(raw_absolute_errors, nan=0.0, posinf=1.0, neginf=0.0)
    
    ae_mae_loss_distribution = np.mean(raw_absolute_errors, axis=(1, 2))
    ae_mse_loss = np.mean(np.square(raw_absolute_errors))

    precision_curve, recall_curve, thresholds = precision_recall_curve(true_anomalies_proxy, ae_mae_loss_distribution)
    f1_scores = (2 * precision_curve[:-1] * recall_curve[:-1]) / (precision_curve[:-1] + recall_curve[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
    anomaly_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5

    predicted_anomalies = (ae_mae_loss_distribution > anomaly_threshold).astype(int)
    precision = precision_score(true_anomalies_proxy, predicted_anomalies, zero_division=0)
    recall = recall_score(true_anomalies_proxy, predicted_anomalies, zero_division=0)
    f1 = f1_score(true_anomalies_proxy, predicted_anomalies, zero_division=0)

    try: ae_roc_auc = roc_auc_score(true_anomalies_proxy, ae_mae_loss_distribution)
    except ValueError: ae_roc_auc = 0.0
        
    pr_auc = auc(recall_curve, precision_curve)
    cm = confusion_matrix(true_anomalies_proxy, predicted_anomalies)

    # --- MOVED ULTIMATE ALIGNMENT FIX UP HERE TO ALLOW FOR PROPER EVALUATION ---
    full_len = len(df_export)
    def align_array(arr, target_len):
        pad_size = target_len - len(arr)
        if pad_size > 0: return np.pad(arr, (pad_size, 0), mode='edge') 
        return arr[:target_len]

    aligned_recon_lat = align_array(final_polished_lat, full_len)
    aligned_recon_lon = align_array(final_polished_lon, full_len)
    aligned_speed = align_array(speed_recon * 3.6, full_len)
    aligned_accel = align_array(accel_recon, full_len)
    aligned_turn = align_array(turn_rate_recon, full_len)
    aligned_ae_loss = align_array(ae_mae_loss_distribution, full_len)

    # ---------------------------------------------------------------------
    # PRINTING MODEL METRICS
    # ---------------------------------------------------------------------
    print("\n" + "="*70)
    print(" BERT4Traj TRANSFORMER RECONSTRUCTION ACCURACY:")
    print("="*70)

    mask_test = y_test_t[:, 3]
    eval_mask = (mask_test == 0.0)
    
    if np.sum(eval_mask) > 0:
        trans_eval_pred = transformer_model.predict(X_test_t, verbose=0)
        
        # 1. Unscale the predicted and true step deltas
        true_unscaled = scaler_y.inverse_transform(y_test_t[:, :3])
        pred_unscaled = scaler_y.inverse_transform(trans_eval_pred[:, :3])
        
        # 2. RECONSTRUCT THE ABSOLUTE GEOGRAPHIC PATHS (Cumulative Sum)
        # We must stack the step updates to build the actual driving tracks
        true_path_lat = np.cumsum(true_unscaled[:, 0])
        true_path_lon = np.cumsum(true_unscaled[:, 1])
        
        pred_path_lat = np.cumsum(pred_unscaled[:, 0])
        pred_path_lon = np.cumsum(pred_unscaled[:, 1])
        
        # 3. Convert absolute path deviations into real world meters
        mean_lat_rad = np.radians(np.mean(df_export['GPSLatitude_Clean'].dropna()))
        
        path_error_lat_m = (true_path_lat - pred_path_lat) * 111000.0
        path_error_lon_m = (true_path_lon - pred_path_lon) * (111000.0 * np.cos(mean_lat_rad))
        
        # 4. Compute the true final Euclidean Overlay Error across the journey
        total_euclid_error = np.sqrt(np.mean(path_error_lat_m[eval_mask]**2 + path_error_lon_m[eval_mask]**2))
        
        # 5. Local frame metrics (Keep these as they are already perfect!)
        mean_lat_rad = np.radians(np.mean(df_export['GPSLatitude_Clean'].dropna()))
        delta_lat_meters = (true_unscaled[:, 0] - pred_unscaled[:, 0]) * 111000.0
        delta_lon_meters = (true_unscaled[:, 1] - pred_unscaled[:, 1]) * (111000.0 * np.cos(mean_lat_rad))
        
        step_rmse_meters = np.sqrt(np.mean(delta_lat_meters[eval_mask]**2 + delta_lon_meters[eval_mask]**2))
        velocity_rmse = np.sqrt(np.mean((true_unscaled[eval_mask, 2] - pred_unscaled[eval_mask, 2])**2))

        print(f"     • SPATIAL DRIFT (RMSE)             : {step_rmse_meters:.2f} meters per frame")
        print(f"     • VELOCITY TRACKING ACCURACY (RMSE): {velocity_rmse:.4f} m/s")
        print(f"     • TOTAL PATH MAP OVERLAY (EUCLIDEAN ERROR)     : {total_euclid_error:.2f} meters")

    print("\n" + "="*70)
    print(" BEHAVIORAL AUTOENCODER ANOMALY DETECTOR PERFORMANCE")
    print("="*70)
    print(f" MSE                 : {ae_mse_loss:.6f}")
    print(f" Threshold           : {anomaly_threshold:.6f}")
    print(f" Predicted Anomalies : {np.sum(predicted_anomalies)} occurrences")
    print(f" True Anomalies      : {np.sum(true_anomalies_proxy)} occurrences")
    print("\n Classification Metrics")
    print(f" Precision           : {precision:.4f}")
    print(f" Recall              : {recall:.4f}")
    print(f" F1 Score            : {f1:.4f}")
    print("\n Confusion Matrix\n", cm)
    print("\n Classification Report\n", classification_report(true_anomalies_proxy, predicted_anomalies, zero_division=0))
    print("="*70)
    # Generate Performance Evaluation Charts
    print("[*] Generating Underfitting / Overfitting Diagnostic Curves...")
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_history.history['loss'], label='Training Loss', color='darkorange', linewidth=2.5)
    plt.plot(t_history.history['val_loss'], label='Validation Loss', color='crimson', linewidth=2.5, linestyle='--')
    plt.title('Transformer Training Curve', fontweight='bold', fontsize=12)
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ae_history.history['loss'], label='Training Loss', color='darkorange', linewidth=2.5)
    plt.plot(ae_history.history['val_loss'], label='Validation Loss', color='crimson', linewidth=2.5, linestyle='--')
    plt.title('Autoencoder Training Curves', fontweight='bold', fontsize=12)
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss (MAE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_export_dir, 'training_diagnostics.png'))
    plt.close()

    print("=== DASHCAM FORENSIC ANALYTICS MODELLING DEVELOPMENT AND EVALUTAION PIPELINE COMPLETELY FINISHED ===")

    # Package the final output for the Dashboard
    final_payload_df = pd.DataFrame({
    'Frame': np.arange(full_len),
    'Raw_Lat': df_export['GPSLatitude_Clean'].values,
    'Raw_Lon': df_export['GPSLongitude_Clean'].values,
    'Recon_Lat': aligned_recon_lat,
    'Recon_Lon': aligned_recon_lon,
    'Speed_Kmh': aligned_speed,
    'Accel_M_S2': aligned_accel,
    'Turn_Rate': aligned_turn,
    'AE_Loss': aligned_ae_loss,
    'GPS_Dropped': df_export['GPS_Missing_Flag'].values
    })
    
    results = {
    "df_features": df_export,
    "df_results": final_payload_df,

    "transformer_rmse": float(step_rmse_meters),
    "velocity_rmse": float(velocity_rmse),
    "euclidean_error": float(total_euclid_error),

    "ae_threshold": float(anomaly_threshold),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(ae_roc_auc),

    "total_anomalies": int(np.sum(predicted_anomalies)),
    "true_anomalies": int(np.sum(true_anomalies_proxy))
    }

    return results

# If running the file directly via command line, use the original hardcoded paths:
if __name__ == "__main__":
    COMBINED_CSV = r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\Dashcam_Forensic_Analytic_Data.csv"
    AUDIT_CSV = r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\Master_Forensic_Audit_Log.csv"
    FEATURES_CSV = r"C:\Users\Owner\Desktop\UMPSA\SEM 6\DSP 2\datasets\new dataset\Preprocessed_Forensic_Features.csv"
    
    run_forensic_pipeline(COMBINED_CSV, AUDIT_CSV, FEATURES_CSV)