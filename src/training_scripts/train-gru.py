from datasetutils import load_lvl_0_dataset
from models.GRUBasedNN import GRUBasedNN
from models.InputScaler import InputScaler

MODEL_NAME = "gru.hdf5"
MODEL_PATH = "trained_models/" + MODEL_NAME
DATA_PATH = "clean_data"
SCALER_PATH = "trained_models/scaler.pkl"
BATCH_SIZE = 1000
TRAIN_SIZE = 450000
DATASET_REPETITIONS = 4

X_train, T_train, Y_train, X_val, T_val, Y_val = load_lvl_0_dataset(
    TRAIN_SIZE, DATA_PATH, DATASET_REPETITIONS)

scaler = InputScaler.load(SCALER_PATH)

X_train, T_train = scaler.transform(X_train, T_train)
X_val, T_val = scaler.transform(X_val, T_val)

model = GRUBasedNN()
model.train(X_train, T_train, Y_train, X_val, T_val, Y_val, MODEL_PATH, BATCH_SIZE)
