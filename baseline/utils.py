import torch

def print_model_info(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            dims = list(param.data.size())
            local_params = 1
            for dim in dims:
                local_params *= dim
            total_params += local_params
            print (name, param.data.size())
    print ("\nTotal Params:{}\n".format(total_params))
