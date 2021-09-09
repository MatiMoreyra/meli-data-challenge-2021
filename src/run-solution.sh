#!/bin/bash

# Extract time series from the dataset:
python3 ./preprocessing/extract-time-series.py

# Generate a supervised learning dataset:
python3 ./preprocessing/generate-sl-dataset.py

# Train all level-0 models:
python3 ./train-all.py

# Train the level-1 ensemble:
python3 ./train-ensemble.py

# Generate the submission file and gzip it:
python3 ./generate-submission.py && gzip ./submission.csv
