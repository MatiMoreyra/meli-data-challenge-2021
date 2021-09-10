from models.LSTMBasedNN import LSTMBasedNN
from models.GRUBasedNN import GRUBasedNN
from models.XGBHuberRegressor import XGBHuberRegressor
from models.XGBMSERegressor import XGBMSERegressor
from models.XGBMultiQuantileRegressor import XGBMultiQuantileRegressor
from models.InputScaler import InputScaler
from datasetutils import load_lvl_0_dataset
import config
from pathlib import Path

# Create clean data directory if not exists
Path(config.TRAINED_MODELS_PATH).mkdir(parents=False, exist_ok=True)

## Train XGB regressors ###
# List of tuples with models and the path argument for their train function.
XGB_MODELS = [
    (XGBHuberRegressor(), config.XGB_HUBBER_REGRESSOR_PATH),
    (XGBMSERegressor(), config.XGB_MSE_REGRESSOR_PATH),
    (XGBMultiQuantileRegressor(), config.XGB_QUANTILE_REGRESSOR_PREFIX)
]

X_train, T_train, Y_train, X_val, T_val, Y_val = load_lvl_0_dataset(
    config.DATA_REPETITIONS_XGB)

scaler = InputScaler()
scaler.fit(X_train, T_train)
scaler.dump(config.SCALER_PATH)

X_train, T_train = scaler.transform(X_train, T_train)
X_val, T_val = scaler.transform(X_val, T_val)

for m in XGB_MODELS:
    model = m[0]
    path = m[1]
    model.train(X_train, T_train, Y_train, X_val, T_val, Y_val, path)

### Train NN models ###
NN_MODELS = [
    (LSTMBasedNN(), config.LSTM_PATH),
    (GRUBasedNN(), config.GRU_PATH)
]

X_train, T_train, Y_train, X_val, T_val, Y_val = load_lvl_0_dataset(
    config.DATA_REPETITIONS_NN)

X_train, T_train = scaler.transform(X_train, T_train)
X_val, T_val = scaler.transform(X_val, T_val)

for m in NN_MODELS:
    model = m[0]
    path = m[1]
    model.train(X_train, T_train, Y_train, X_val, T_val, Y_val, path)
