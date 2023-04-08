import numpy as np

from ._heuristic import constraints_accuracy, objective_node_accuracy, overall_accuracy  # noqa: F401
from config.vars import CLASSIFICATION_ERROR_DECIMALS, REGRESSION_ERROR_DECIMALS


def accuracy(y_pred, y_true):
    error = (y_pred == y_true) * 1.0
    error = np.mean(error)
    return round(error, CLASSIFICATION_ERROR_DECIMALS)


def eval_categorical(y_pred, y_true):
    return accuracy(y_pred.argmax(-1), y_true.argmax(-1))


def mse(y_pred, y_true):
    error = (y_pred - y_true)**2
    error = np.mean(error)
    return round(error, REGRESSION_ERROR_DECIMALS)


def mae(y_pred, y_true):
    error = np.abs((y_pred - y_true)).mean()
    return round(error, REGRESSION_ERROR_DECIMALS)


def masked_mae(y_pred, y_true):
    mask = y_true != 0
    error = np.abs(y_pred - y_true)[mask].mean()
    return round(error, REGRESSION_ERROR_DECIMALS)


def masked_mse(y_pred, y_true):
    mask = y_true != 0
    error = (y_pred - y_true)**2
    error = error[mask].mean()
    return round(error, REGRESSION_ERROR_DECIMALS)


def dual_objective(y_pred, inputs):
    for inp in inputs:
        if inp.name == 'adj':
            adj = inp.data.numpy()
        elif inp.name == 'A':
            weights = inp.data.numpy()

    return round(constraints_accuracy(y_pred, weights, adj), CLASSIFICATION_ERROR_DECIMALS)


def mask_fn(pred, truth):
    tp = ((pred > 0.5) * (truth > 0.5)) * 1.0
    fp = ((pred > 0.5) * (truth < 0.5)) * 1.0
    fn = ((pred < 0.5) * (truth > 0.5)) * 1.0

    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)

    if tp + fp + fn == 0:
        return 1.

    if tp == 0:
        return 0.

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_1 = 2.0 * precision * recall / (precision + recall)

    return round(f_1, CLASSIFICATION_ERROR_DECIMALS)


def eval_one(pred, truth):
    error = np.argmax(pred, -1) == np.argmax(truth, -1)
    error = np.mean(error)
    return round(error, CLASSIFICATION_ERROR_DECIMALS)
