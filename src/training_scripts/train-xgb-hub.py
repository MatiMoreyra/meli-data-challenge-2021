from models.XGBHubberRegressor import XGBHubberRegressor
from models.InputScaler import InputScaler
from datasetutils import load_lvl_0_dataset

TRAIN_SIZE = 450000
DATA_PATH = "clean_data"
OUTPUT_PATH = "trained_models/xgb-hub.json"
SCALER_PATH = "trained_models/scaler.pkl"
DATASET_REPETITIONS = 1

X_train, T_train, Y_train, X_val, T_val, Y_val = load_lvl_0_dataset(TRAIN_SIZE, "clean_data", DATASET_REPETITIONS)

scaler = InputScaler.load(SCALER_PATH)

X_train, T_train = scaler.transform(X_train, T_train)
X_val, T_val = scaler.transform(X_val, T_val)

model = XGBHubberRegressor()
model.train(X_train, T_train, Y_train, X_val, T_val, Y_val, OUTPUT_PATH)
