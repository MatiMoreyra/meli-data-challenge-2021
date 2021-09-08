
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
from models.InputScaler import InputScaler
from models.LSTMBasedNN import LSTMBasedNN
from models.GRUBasedNN import GRUBasedNN
from models.XGBHubberRegressor import XGBHubberRegressor
from models.XGBMSERegressor import XGBMSERegressor
from models.XGBMultiQuantileRegressor import XGBMultiQuantileRegressor

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

DATA_PATH = "clean_data"
DATASET_DIRECTORY = "/home/mati/repos/ml/meli-2021/dataset"
ENSEMBLE_PATH = "trained_models/lstm.hdf5"
BATCH_SIZE = 500
SCALER_PATH = "trained_models/scaler.pkl"
# List of tuples with models and the argument for its load(path) function.
MODELS = [(LSTMBasedNN(), "trained_models/lstm.hdf5"),
          (GRUBasedNN(), "trained_models/gru.hdf5"),
          (XGBHubberRegressor(), "trained_models/xgb-hub.json"),
          (XGBMSERegressor(), "trained_models/xgb-mse.json"),
          (XGBMultiQuantileRegressor(), "trained_models/xgb")]

# Build submission file
df_test = pd.read_csv(DATASET_DIRECTORY + '/test_data.csv')

targets_skus = df_test["sku"].to_numpy() - 1
targets = df_test["target_stock"].to_numpy()

print("Targets:" + str(targets.shape[0]))

x_input = np.load(DATA_PATH + "/series.npy")[:,-29:,1:]
series = None
t_input = np.ones((x_input.shape[0],1))
t_input[targets_skus,0]=targets

scaler = InputScaler.load(SCALER_PATH)

x_input, t_input = scaler.transform(x_input, t_input)

lvl_0_outputs = list()

for m in MODELS:
    model = m[0]
    model.load(m[1])
    lvl_0_outputs.append(model.predict(x_input, t_input))

X = np.hstack(lvl_0_outputs)

model = tf.keras.models.load_model('trained_models/ensemble.hdf5',compile=False)

y_pred = model.predict(X, batch_size=BATCH_SIZE)

predictions = y_pred[targets_skus,:]

np.savetxt("submission.csv",predictions,fmt="%.4f",delimiter=",")