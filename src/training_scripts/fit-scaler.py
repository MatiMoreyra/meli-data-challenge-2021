from datasetutils import load_lvl_0_dataset
from models.InputScaler import InputScaler

OUTPUT_PATH = "trained_models/scaler.pkl"
DATA_PATH = "clean_data"
BATCH_SIZE = 1000
TRAIN_SIZE = 450000
DATASET_REPETITIONS = 2

X_train, T_train, _, _, _, _ = load_lvl_0_dataset(
    TRAIN_SIZE, DATA_PATH, DATASET_REPETITIONS)

scaler = InputScaler()
scaler.fit(X_train, T_train)
scaler.dump(OUTPUT_PATH)
