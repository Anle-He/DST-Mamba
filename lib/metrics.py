import numpy as np

import torch
import torch.nn as nn


def select_loss(loss_name):

    if loss_name.upper() == 'MAE':
        return MaskedMAELoss
    elif loss_name.upper() == 'MSE':
        return nn.MSELoss
    # TODO: support order-robust training.
    else:
        raise ValueError(f'Invalid loss: {loss_name}')


def MaskedMAELoss():

    def _get_name(self):
        return self.__class__.__name__


    def __call__(self, preds, labels, null_val=np.nan):
        return masked_mae(preds, labels, null_val)


def masked_mae(preds, labels, null_val):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def MaskedMSELoss():
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=np.nan):
        return masked_mse(preds, labels, null_val)


def masked_mse(preds, labels, null_val=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)

    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


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