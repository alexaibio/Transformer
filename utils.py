

def vizualize_model(model):
    mdl_keys = model.state_dict().keys()
    print('\n>>> Model defined in code:')
    [print(k) for k in mdl_keys]