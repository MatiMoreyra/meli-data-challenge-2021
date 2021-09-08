import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from models.EnsembleModel import create_ensemble_model
from models.LSTMBasedNN import LSTMBasedNN
from models.GRUBasedNN import GRUBasedNN
from models.XGBHubberRegressor import XGBHubberRegressor
from models.XGBMSERegressor import XGBMSERegressor
from models.XGBMultiQuantileRegressor import XGBMultiQuantileRegressor
from models.InputScaler import InputScaler
from rps import rps
from datasetutils import load_lvl_1_dataset
import config

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# List of tuples with models and the argument for its load(path) function.
MODELS = [(LSTMBasedNN(), config.LSTM_PATH),
          (GRUBasedNN(), config.GRU_PATH),
          (XGBHubberRegressor(), config.XGB_HUBBER_REGRESSOR_PATH),
          (XGBMSERegressor(), config.XGB_MSE_REGRESSOR_PATH),
          (XGBMultiQuantileRegressor(), config.XGB_QUANTILE_REGRESSOR_PREFIX)]

X_train, T_train, Y_train, X_val, T_val, Y_val = load_lvl_1_dataset(config.DATA_REPETITIONS_NN)

scaler = InputScaler.load(config.SCALER_PATH)

X_train, T_train = scaler.transform(X_train, T_train)
X_val, T_val = scaler.transform(X_val, T_val)

print(X_train.shape)
print(X_val.shape)

lvl_0_outputs_train = list()
lvl_0_outputs_val = list()

for m in MODELS:
    model = m[0]
    model.load(m[1])
    lvl_0_outputs_train.append(model.predict(X_train, T_train))
    lvl_0_outputs_val.append(model.predict(X_val, T_val))

X_train = np.hstack(lvl_0_outputs_train)
X_val = np.hstack(lvl_0_outputs_val)

model = create_ensemble_model(X_train.shape[1])

opt = Adam(learning_rate=0.001)
model.compile(loss=rps, optimizer=opt)
model.summary()

reduce_lr_loss = ReduceLROnPlateau(
    monitor='loss', factor=0.5, patience=7, verbose=1, epsilon=1e-4, mode='min')

# No need for early stopping.
try:
    model.fit(X_train, Y_train, batch_size=config.DEFAULT_BATCH_SIZE, epochs=200,
              verbose=True, callbacks=[reduce_lr_loss], validation_data=(X_val, Y_val), shuffle=True)
except KeyboardInterrupt:
    print("Stopped training.")

model.save(config.ENSEMBLE_MODEL_PATH)

y_pred = model.predict(X_val, batch_size=config.DEFAULT_BATCH_SIZE)

print("RPS val:" + str(rps(y_pred.astype(np.float32), Y_val.astype(np.float32))))
