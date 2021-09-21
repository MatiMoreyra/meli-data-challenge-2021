import os, sys
# Append the src directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from datasetutils import load_lvl_0_dataset
from models.InputScaler import InputScaler

X_train, T_train, _, _, _, _ = load_lvl_0_dataset()

scaler = InputScaler()
scaler.fit(X_train, T_train)
scaler.dump(config.SCALER_PATH)
