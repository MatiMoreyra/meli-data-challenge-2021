import os
import numpy as np
import config

DATASET_REPETITIONS = 4
SKU_COLUMN_INDEX = 0
SALES_COUMN_INDEX = 1

numpy_series = None

if os.path.exists(config.TIME_SERIES_PATH):
    numpy_series = np.load(config.TIME_SERIES_PATH)
else:
    print("Error loading {}, please run extract-time-series.py first.".format(config.TIME_SERIES_PATH))
    exit(1)

# Function that generates a dataset for supervised learning.
def generate_supervised_lerning_dataset(numpy_series, seed):
    total_sales = np.sum(numpy_series[:, -30:, SALES_COUMN_INDEX], axis=1).reshape(-1, 1)

    rng = np.random.default_rng(seed)

    # Generate target stock as a random integer between 1 and the total sold quantity of the last 28 days
    target_stock = rng.integers(1, np.clip(
        total_sales + 1, 2, None), size=(numpy_series.shape[0], 1))
    target_stock = target_stock.astype(int)

    # Compute how many days it took to sold out the target stock.
    cum_sales = np.cumsum(numpy_series[:, -30:, SALES_COUMN_INDEX], axis=1)
    stock_days = np.argmax(cum_sales >= target_stock,
                           axis=1).reshape(-1, 1) + 1

    one_hot_stock_days = np.zeros((stock_days.shape[0], 30))
    one_hot_stock_days[np.arange(stock_days.shape[0]),
                       stock_days[:, 0] - 1] = 1

    return numpy_series[:, :-30, :], target_stock, one_hot_stock_days

# Shuffle the data.
rng = np.random.default_rng(2021)
rng.shuffle(numpy_series)

# Filter invalid skus
numpy_series = numpy_series[np.count_nonzero(numpy_series[:,:-30,SKU_COLUMN_INDEX],axis=1) >= 1,:,:]
numpy_series = numpy_series[np.count_nonzero(numpy_series[:,-30:,SKU_COLUMN_INDEX],axis=1) == 30,:,:]
total_sales = np.sum(numpy_series[:, -30:, SALES_COUMN_INDEX], axis=1)
numpy_series = numpy_series[total_sales > 0, :, :]

# On each iteration we generate different target stock and stock day values.
# This can be thought as a kind of data augmentation.
for i in range(DATASET_REPETITIONS):
    S, T, Y = generate_supervised_lerning_dataset(numpy_series, i + 2000)
    np.savez(config.SL_DATASET_TEMPLATE.format(i), S=S, T=T, Y=Y)