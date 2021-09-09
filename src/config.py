from pathlib import Path


def absolute_path(path):
    src_path = str(Path(__file__).parent.resolve())
    return src_path + "/" + path


# Paths
DATASET_DIRECTORY = absolute_path("../dataset")
CLEAN_DATA_PATH = absolute_path("clean_data")
TIME_SERIES_PATH = absolute_path("clean_data/series.npy")
TRAINED_MODELS_PATH = absolute_path("trained_models")
SCALER_PATH = absolute_path("trained_models/scaler.pkl")
GRU_PATH = absolute_path("trained_models/gru.hdf5")
LSTM_PATH = absolute_path("trained_models/lstm.hdf5")
XGB_MSE_REGRESSOR_PATH = absolute_path("trained_models/xgb-mse.json")
XGB_HUBBER_REGRESSOR_PATH = absolute_path("trained_models/xgb-hub.json")
XGB_QUANTILE_REGRESSOR_PREFIX = absolute_path("trained_models/xgb-")
ENSEMBLE_MODEL_PATH = absolute_path("trained_models/ensemble.hdf5")
SL_DATASET_TEMPLATE = absolute_path("clean_data/dataset-{}.npz")


# Create directories if not exist
Path(TRAINED_MODELS_PATH).mkdir(parents=False, exist_ok=True)
Path(CLEAN_DATA_PATH).mkdir(parents=False, exist_ok=True)

# Training set size.
LVL_0_TRAIN_SIZE = 450000
LVL_1_TRAIN_SIZE = 105000

# Default batch size for every model
DEFAULT_BATCH_SIZE = 1000

# Dataset repetitions (data augmentation).
# XGB regressors seem to work better without data augmentation.
DATA_REPETITIONS_XGB = 1

# NN models seem to work better with 3 dataset repetitions.
DATA_REPETITIONS_NN = 3
