import numpy as np
import torch

def nmse_loss(y_pred, y_true):
    # 自动 reshape 成 1D
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        mse = np.mean((y_pred - y_true) ** 2)
        norm = np.mean(y_true ** 2)
        nmse = mse / (norm + 1e-8)
    elif isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        mse = torch.mean((y_pred - y_true) ** 2)
        norm = torch.mean(y_true ** 2)
        nmse = mse / (norm + 1e-8)
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")

    return nmse
