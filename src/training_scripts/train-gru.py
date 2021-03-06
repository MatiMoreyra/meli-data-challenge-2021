import os, sys
# Append the src directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from datasetutils import load_lvl_0_dataset
from models.GRUBasedNN import GRUBasedNN
from models.InputScaler import InputScaler


X_train, T_train, Y_train, X_val, T_val, Y_val = load_lvl_0_dataset(config.DATA_REPETITIONS_NN)

scaler = InputScaler.load(config.SCALER_PATH)

X_train, T_train = scaler.transform(X_train, T_train)
X_val, T_val = scaler.transform(X_val, T_val)

model = GRUBasedNN()
model.train(X_train, T_train, Y_train, X_val, T_val, Y_val, config.GRU_PATH, config.DEFAULT_BATCH_SIZE)
