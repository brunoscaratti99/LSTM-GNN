def reset_weights(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()