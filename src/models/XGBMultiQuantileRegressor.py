import xgboost as xgb
import numpy as np

def xgb_quantile_eval(quantile):
    q = quantile

    def _xgb_quantile_eval(preds, dmatrix):
        labels = dmatrix.get_label()
        return ('q{}_loss'.format(q),
                np.nanmean((preds >= labels) * (1 - q) * (preds - labels) +
                           (preds < labels) * q * (labels - preds)))
    return _xgb_quantile_eval


def xgb_quantile_obj(quantile):
    q = quantile

    def _xgb_quantile_obj(preds, dmatrix):
        try:
            assert 0 <= q <= 1
        except AssertionError:
            raise ValueError("Quantile value must be float between 0 and 1.")

        labels = dmatrix.get_label()
        errors = preds - labels

        left_mask = errors < 0
        right_mask = np.logical_not(left_mask)

        grad = -q * left_mask + (1 - q) * right_mask
        hess = np.ones_like(preds)

        return grad, hess
    return _xgb_quantile_obj


class XGBMultiQuantileRegressor():
    QUANTILES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    def __init__(self):
        self.regressors = list()

    def train(self, X_train, T_train, Y_train, X_test, T_test, Y_test, output_prefix):

        X_train = np.concatenate(
            (X_train.reshape(X_train.shape[0], -1), T_train), axis=1)
        Y_train = np.argmax(Y_train, 1).reshape(-1, 1)

        X_test = np.concatenate(
            (X_test.reshape(X_test.shape[0], -1), T_test), axis=1)
        Y_test = np.argmax(Y_test, 1).reshape(-1, 1)

        train = xgb.DMatrix(X_train, Y_train)
        test = xgb.DMatrix(X_test, Y_test)

        params = {
            "learning_rate": 1,
            "max_depth": 5,
            "subsample": 0.5,
            "colsample_bytree": 0.8,
            "tree_method": 'gpu_hist',
            "min_child_weight": 0,
            "gamma": 0,
            "gpu_id": 0,
            'disable_default_eval_metric': 1,
            "seed": 2021
        }

        for q in XGBMultiQuantileRegressor.QUANTILES:
            # training, we set the early stopping rounds parameter
            model = xgb.train(params,
                              train, evals=[(train, "train"),
                                            (test, "validation")],
                              num_boost_round=15000, early_stopping_rounds=150, obj=xgb_quantile_obj(q), feval=xgb_quantile_eval(q))

            path = output_prefix + "-q{}.json".format(q)
            model.save_model(path)

            model = xgb.XGBRegressor()
            model.load_model(path)

            self.regressors.append(model)

    def load(self, file_prefix):
        for q in XGBMultiQuantileRegressor.QUANTILES:
            model = xgb.XGBRegressor()
            path = file_prefix + "-q{}.json".format(q)
            model.load_model(path)
            self.regressors.append(model)

    def predict(self, X, T):
        outputs = list()
        input = np.concatenate((X.reshape(X.shape[0], -1), T), axis=1)
        for model in self.regressors:
            outputs.append(model.predict(input).reshape(-1, 1))
        return np.hstack(outputs)

    def evaluate(self, X, T, Y):
        Y_pred = self.model.predict(np.concatenate(
            (X.reshape(X.shape[0], -1), T), axis=1))
        return np.mean(np.abs(Y_pred - Y))
