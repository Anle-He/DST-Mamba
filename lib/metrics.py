import numpy as np

import torch
import torch.nn as nn


def select_loss(loss_name):

    if loss_name == 'MAE':
        return nn.L1Loss
    elif loss_name == 'MSE':
        return nn.MSELoss
    elif loss_name == 'HUBER':
        return nn.HuberLoss
    # TODO: support order-robust training.
    else:
        raise ValueError(f'Invalid loss: {loss_name}')


def MSE_MAE(y_true, y_pred):
    
    return (
        MSE(y_true, y_pred),
        MAE(y_true, y_pred),
    )


def RMSE_MAE_MAPE(y_true, y_pred):

    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )


def MSE(y_true, y_pred):

    with np.errstate(divide="ignore", invalid="ignore"):

        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)

        return mse
    

def MAE(y_true, y_pred):

    with np.errstate(divide="ignore", invalid="ignore"):

        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)

        return mae


def RMSE(y_true, y_pred):

    with np.errstate(divide="ignore", invalid="ignore"):

        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))

        return rmse
    

def MAPE(y_true, y_pred, null_val=0):

    with np.errstate(divide="ignore", invalid="ignore"):

        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)

        mask = mask.astype("float32")
        mask /= np.mean(mask)

        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)

        return np.mean(mape) * 100