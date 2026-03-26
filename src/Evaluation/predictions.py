import torch
from Data.preprocessing import inverse_log1p_zscore, apply_log1p_zscore



def prediction(model, X_input, mean=None, std=None):
    with torch.no_grad():
        pred = model(X_input.to(model.device)).detach().cpu()
        
    if mean == None:
        return pred
    return inverse_log1p_zscore(pred, mean, std)