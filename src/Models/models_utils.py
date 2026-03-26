def reset_weights(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()