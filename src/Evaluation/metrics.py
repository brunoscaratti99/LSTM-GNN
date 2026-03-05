import torch



def safe_r2(y_true, y_pred, eps=1e-8):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + eps))


def safe_mape(y_true, y_pred, eps=1e-3):
    return (torch.abs(y_pred - y_true) / (torch.abs(y_true) + eps)).mean() * 100.0


