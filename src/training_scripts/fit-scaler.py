import os
import sys
from datasetutils import load_lvl_0_dataset
from models.InputScaler import InputScaler

sys.path.append(os.path.abspath('./'))
import config

X_train, T_train, _, _, _, _ = load_lvl_0_dataset()

scaler = InputScaler()
scaler.fit(X_train, T_train)
scaler.dump(config.SCALER_PATH)
