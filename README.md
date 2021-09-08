# Meli Data Challenge 2021
My solution for the Meli Data Challenge 2021

## The Model
My final model is an ensemble combining recurrent neural networks and XGBoost regressors.
Neural networks are trained to predict the stock days probability distribution using the RPS as loss directly.
XGBoost regressors are trained to predict stock days using different losses, here the intuition behind this:
  - MSE loss: the regresor trained with this loss will output values close to the expected mean.
  - Pseudo-Huber loss: an alternative for the MAE loss, this regressor outputs values close to the expected median.
  - Quantile loss: 11 regressors are trained using a quantile loss with alpha 0, 0.1, 0.2, ..., 1. This helps to build the final probability distribution.

The outputs of all these level-0 models are concatenated to train a feedforward neural network with the RPS as loss function.

![diagram](diagram.png)

The last 30 days of the train dataset are used to generate the labels and the target stock input. The remaining 29 days are used to generate the time series input.

Once all models are trained, the last 29 days of the train dataset and the provided target stock values are used to generate the submission.

## How to run the solution

### Requirements
  - TensorFlow v2.
  - Pandas.
  - Numpy.
  - Scikit-learn.

CUDA drivers and a CUDA-compatible GPU is required (I didn't have the time to test this on a cpu).

Some scripts require up to 30GB of RAM (again, I didn't have the time to implement a more memory-friendly solution).

### Running the scripts

Go to the `src` directory:
```
cd src
```
Extract time series from the dataset:
```
python3 ./preprocessing/extract-time-series.py
```
Generate a supervised learning dataset:
```
python3 ./preprocessing/generate-sl-dataset.py
```
Train all level-0 models:
```
python3 ./train-all.py
```
Train the level-1 ensemble:
```
python3 ./train-ensemble.py
```
Generate the submission file and gzip it:
```
python3 ./generate-submission.py && gzip ./submission.csv
```
