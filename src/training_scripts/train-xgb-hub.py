import os
import sys
from models.XGBHuberRegressor import XGBHuberRegressor
from models.InputScaler import InputScaler
from datasetutils import load_lvl_0_dataset
sys.path.append(os.path.abspath('./'))
import config

X_train, T_train, Y_train, X_val, T_val, Y_val = load_lvl_0_dataset(config.DATA_REPETITIONS_XGB)

scaler = InputScaler.load(config.SCALER_PATH)

X_train, T_train = scaler.transform(X_train, T_train)
X_val, T_val = scaler.transform(X_val, T_val)

model = XGBHuberRegressor()
model.train(X_train, T_train, Y_train, X_val, T_val, Y_val, config.XGB_HUBBER_REGRESSOR_PATH)
