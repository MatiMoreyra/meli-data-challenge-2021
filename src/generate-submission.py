
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from models.InputScaler import InputScaler
from models.LSTMBasedNN import LSTMBasedNN
from models.GRUBasedNN import GRUBasedNN
from models.XGBHuberRegressor import XGBHuberRegressor
from models.XGBMSERegressor import XGBMSERegressor
from models.XGBMultiQuantileRegressor import XGBMultiQuantileRegressor

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

BATCH_SIZE = 500

# List of tuples with models and the argument for its load(path) function.
MODELS = [(LSTMBasedNN(), config.LSTM_PATH),
          (GRUBasedNN(), config.GRU_PATH),
          (XGBHuberRegressor(), config.XGB_HUBBER_REGRESSOR_PATH),
          (XGBMSERegressor(), config.XGB_MSE_REGRESSOR_PATH),
          (XGBMultiQuantileRegressor(), config.XGB_QUANTILE_REGRESSOR_PREFIX)]

# Build submission file
df_test = pd.read_csv(config.DATASET_DIRECTORY + '/test_data.csv')

targets_skus = df_test["sku"].to_numpy() - 1
targets = df_test["target_stock"].to_numpy()

print("Targets:" + str(targets.shape[0]))

x_input = np.load(config.TIME_SERIES_PATH)[:,-29:,1:]
series = None
t_input = np.ones((x_input.shape[0],1))
t_input[targets_skus,0]=targets

scaler = InputScaler.load(config.SCALER_PATH)

x_input, t_input = scaler.transform(x_input, t_input)

lvl_0_outputs = list()

for m in MODELS:
    model = m[0]
    model.load(m[1])
    lvl_0_outputs.append(model.predict(x_input, t_input))

X = np.hstack(lvl_0_outputs)

model = tf.keras.models.load_model(config.ENSEMBLE_MODEL_PATH,compile=False)

y_pred = model.predict(X, batch_size=BATCH_SIZE)

predictions = y_pred[targets_skus,:]

np.savetxt("submission.csv",predictions,fmt="%.4f",delimiter=",")
