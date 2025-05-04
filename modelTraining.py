import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1
import joblib
import pickle

# --- Load dataset ---
df = pd.read_csv("Final_Dataset_2013_2022.csv")

df['index'] = df['index'] + 1

# --- Keep necessary columns ---
target_col = ['IRI_VN']
metadata_cols = ['geometry_paths', 'ROUTE_ID']
feature_cols = [col for col in df.columns if col not in target_col + metadata_cols + ['index']]

# --- Create sequences ---
def create_sequences(df, input_features, target_col, window_size=8):
    dataX, dataY, meta_info = [], [], []
    for i in range(len(df) - window_size):
        x_seq = df[input_features].iloc[i:i+window_size].values
        y_seq = df[target_col].iloc[i+window_size].values
        meta_seq = df[metadata_cols].iloc[i+window_size] 
        dataX.append(x_seq)
        dataY.append(y_seq)
        meta_info.append(meta_seq)
    return np.array(dataX), np.array(dataY), pd.DataFrame(meta_info)

X, y, meta = create_sequences(df, feature_cols, target_col)


# --- Spliting the data ---
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(X, y, meta, test_size=0.2, random_state=42)


feature_cols = ['AADT_VN', 'BEGIN_POIN', 'COUNTY_COD', 'END_POINT', 'IS_IMPROVED',
                'SPEED_LIMI', 'THROUGH_LA', 'YEAR_RECOR', 'curval', 'tmiles', 'tons', 'value']


flattened_rows = X_test.reshape(-1, X_test.shape[-1]) 
X_test_original_format_df = pd.DataFrame(flattened_rows, columns=feature_cols)
X_test_original_format_df.reset_index(drop=True, inplace=True)
X_test_original_format_df.to_pickle("X_test_original_flat.pkl")
print(X_test_original_format_df.head())


# --- Standardize ---
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_shape = X_train.shape
y_train_shape = y_train.shape

X_train_scaled = x_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train_shape)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train_shape)


# --- Define TCN model ---
def build_tcn_model(input_shape, output_units):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=128, kernel_size=2, padding='causal', activation='relu', dilation_rate=1, kernel_regularizer = l1(0.00001)))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=96, kernel_size=2, padding='causal', activation='relu', dilation_rate=2, kernel_regularizer = l1(0.00001)))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=96, kernel_size=3, padding='causal', activation='relu', dilation_rate=4, kernel_regularizer = l1(0.00001)))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=96, kernel_size=3, padding='causal', activation='relu', dilation_rate=4, kernel_regularizer = l1(0.00001)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l1(0.00001)))
    model.add(Dropout(0.1))
    model.add(Dense(output_units))
    model.compile(optimizer=Adam(learning_rate=0.00030508), loss='mse')
    return model

# --- Train TCN model ---
model = build_tcn_model((X_train.shape[1], X_train.shape[2]), y.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train_scaled, y_train_scaled, epochs=200, batch_size=16, validation_split=0.1, callbacks=[early_stopping], verbose=1)

# --- Save model and scaler ---
model.save("tcn_model.h5")
joblib.dump(x_scaler, "x_scaler.save")
joblib.dump(y_scaler, "y_scaler.save")