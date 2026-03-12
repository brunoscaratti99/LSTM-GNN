from Data.preprocessing import inverse_log1p_zscore, apply_log1p_zscore



def prediction(model, X_input, mean, std):
    pred = model(X_input.to(model.device)).detach().cpu()
    pred_denormalized = inverse_log1p_zscore(pred, mean, std)