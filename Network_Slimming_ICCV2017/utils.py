import numpy as np
from collections import OrderedDict
import torch

from prettytable import PrettyTable

def compare_parameters(model1, model2):
    table = PrettyTable(["Modules", "Pre-Pruning", "Post-Pruning"])
    total_params1 = 0
    totla_params2 = 0
    for (name, parameter1), (_, parameter2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not parameter1.requires_grad: continue
        param = parameter1.numel()
        param2 = parameter2.numel()
        table.add_row([name, param, param2])
        total_params1+=param
        totla_params2+=param2
    print(table)
    print(f"Total Trainable Params; Pre-Pruning: {total_params1}, Post-Pruning: {totla_params2}")
    return total_params1
    

def extract_prune(weight):
    gamma = np.array(weight.cpu().detach())
    # import pdb;pdb.set_trace()
    size = gamma.shape[0]//4
    return np.argpartition(gamma,-size)[-size:]

def get_prune_idx(model_dict):
    pruned_idx = OrderedDict()
    for idx, (key, value) in enumerate(list(model_dict.items())):
        # print(idx, key)
        # import pdb;pdb.set_trace()
        if 'bn' in key and 'weight' in key:
            # import pdb;pdb.set_trace()
            top_idx = extract_prune(value)
            pruned_idx[idx] = top_idx

    return pruned_idx

def get_new_weights(model_weights):
    pruned_idx = get_prune_idx(model_weights)
    last_idx = None
    for idx, (key, value) in enumerate(pruned_idx.items()):
        (layer1, conv_weight) = list(model_weights.items())[key-2]
        (layer2, conv_bias) = list(model_weights.items())[key-1]
        (layer3, bn_weight) = list(model_weights.items())[key]
        (layer4, bn_bias) = list(model_weights.items())[key+1]
        
        conv_weight.data = torch.index_select(conv_weight, 0, torch.tensor(value))
        conv_bias.data = torch.index_select(conv_bias, 0, torch.tensor(value))
        bn_weight.data = torch.index_select(bn_weight, 0, torch.tensor(value))
        bn_bias.data = torch.index_select(bn_bias, 0, torch.tensor(value))

        if last_idx is not None:
            if last_idx.shape[0] < conv_weight.data.shape[1]:
                conv_weight.data = torch.index_select(conv_weight, 1, torch.tensor(last_idx))
        last_idx = value

    value = list(pruned_idx.items())[-1][1]
    for layer, param in model_weights.items():
        if layer == 'classifier.0.weight' or layer == 'fc.weight':
            param = torch.index_select(param, 1, torch.tensor(value))
            model_weights[layer] = param
    
    return model_weights
