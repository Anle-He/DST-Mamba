import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from lib.scalers import StandardScaler
from lib.utils import print_log


def select_dataloader(task):

    if task.upper() == 'STF':
        return build_STF_dataloader
    elif task.upper() == 'LTSF':
        return build_LTSF_dataloader
    else:
        raise ValueError(f'{task} dataloader has not been implemented yet')


def build_STF_dataloader(
        data_dir,
        batch_size=32,
        in_steps=12,
        out_steps=12,
        x_tod=False,
        x_dow=False,
        y_tod=False,
        y_dow=False,
        log=None):
    
    data = np.load(os.path.join(data_dir, f'processed_data.npz'))['data'].astype(np.float32)
    index = np.load(os.path.join(data_dir, f'index_in{in_steps}_out{out_steps}.npz'))

    x_features = [0]
    if x_tod:
        x_features.append(1)
    if x_dow:
        x_features.append(2)

    y_features = [0]
    if y_tod:
        y_features.append(1)
    if y_dow:
        y_features.append(2)

    train_index = index['train'] # [num_samples, 3]
    val_index = index['val']
    test_index = index['test']

    # Interative
    x_train = np.stack([data[idx[0] : idx[1]] for idx in train_index])[..., x_features]
    y_train = np.stack([data[idx[1] : idx[2]] for idx in train_index])[..., y_features]
    x_val = np.stack([data[idx[0] : idx[1]] for idx in val_index])[..., x_features]
    y_val = np.stack([data[idx[1] : idx[2]] for idx in val_index])[..., y_features]
    x_test = np.stack([data[idx[0] : idx[1]] for idx in test_index])[..., x_features]
    y_test = np.stack([data[idx[1] : idx[2]] for idx in test_index])[..., y_features]


    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler


def build_LTSF_dataloader(
        data_dir,
        batch_size,
        in_steps=96,
        out_steps=96,
        x_tod=False,
        x_dow=False,
        y_tod=False,
        y_dow=False,
        log=None):

    data = np.load(os.path.join(data_dir, f'processed_data.npz'))['data'].astype(np.float32)
    index = np.load(os.path.join(data_dir, f'index_in{in_steps}_out{out_steps}.npz'))

    # Add timestamp features
    x_features = [0]
    if x_tod:
        x_features.append(1)
    if x_dow:
        x_features.append(2)

    y_features = [0]
    if y_tod:
        y_features.append(1)
    if y_dow:
        y_features.append(2)


    # (num_samples, 3)
    train_index = index['train']
    val_index = index['val']
    test_index = index['test']

    len_train = train_index[-1][1]
    scaler = StandardScaler(
        mean=data[:len_train, :, 0].mean(axis=0), 
        std=data[:len_train, :, 0].std(axis=0)
    )
    data[..., 0] = scaler.transform(data[..., 0])


    # Iterative
    x_train = np.stack([data[idx[0] : idx[1]] for idx in train_index])[..., x_features]
    y_train = np.stack([data[idx[1] : idx[2]] for idx in train_index])[..., y_features]
    x_val = np.stack([data[idx[0] : idx[1]] for idx in val_index])[..., x_features]
    y_val = np.stack([data[idx[1] : idx[2]] for idx in val_index])[..., y_features]
    x_test = np.stack([data[idx[0] : idx[1]] for idx in test_index])[..., x_features]
    y_test = np.stack([data[idx[1] : idx[2]] for idx in test_index])[..., y_features]

    print_log(f'Trainset:\tx-{x_train.shape}\ty-{y_train.shape}', log=log)
    print_log(f'Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}', log=log)
    print_log(f'Testset:\tx-{x_test.shape}\ty-{y_test.shape}', log=log)
    print_log('INFO: Using scaled X and Y', log=log)


    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), 
        torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), 
        torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), 
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler