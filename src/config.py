# Global configuration.

# Paths
DATASET_DIRECTORY = "dataset"
CLEAN_DATA_PATH = "clean_data"
TIME_SERIES_PATH = "clean_data/series.npy"
SCALER_PATH = "trained_models/scaler.pkl"
GRU_PATH = "trained_models/gru.hdf5"
LSTM_PATH = "trained_models/lstm.hdf5"
XGB_MSE_REGRESSOR_PATH = "trained_models/xgb-mse.json"
XGB_HUBBER_REGRESSOR_PATH = "trained_models/xgb-hub.json"
XGB_QUANTILE_REGRESSOR_PREFIX= "trained_models/xgb-"
ENSEMBLE_MODEL_PATH = "trained_models/ensemble.hdf5"
SL_DATASET_TEMPLATE = "clean_data/dataset-{}.npz"

# Training set size.
LVL_0_TRAIN_SIZE = 450000
LVL_1_TRAIN_SIZE = 105000

# Default batch size for every model
DEFAULT_BATCH_SIZE = 1000

# Dataset repetitions (data augmentation).
# XGB regressors seem to work better without data augmentation.
DATA_REPETITIONS_XGB = 1

# NN models seem to work better with 2 dataset repetitions.
DATA_REPETITIONS_NN = 4
