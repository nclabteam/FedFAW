import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import re
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def filter_params_to_merge(param_names, exclude_param_regex):
    params_to_merge = []
    for name in param_names:
        valid = not any([re.match(patt, name) for patt in exclude_param_regex])
        if valid:
            params_to_merge.append(name)
    return params_to_merge

def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any([re.match(patt, name) for patt in include_patterns])
        valid_type = not include_type or any([isinstance(module, md_cls) for md_cls in include_type])
        if valid_type and valid_name:
            modules[name] = module
    return modules

def compute_gram(model, dataset, device='cuda', n_step=100):
    """
    Computes Gram matrices for linear layers.
    """
    model.to(device)
    model.eval()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    grams = {} # gram matrices for each linear layer inputs
    xn = {} # number of examples used for computing gram

    def get_gram(name):
        def hook(module, input, output):
            x = input[0].detach() # [b,t,h] or [b,h]
            x = x.view(-1, x.size(-1))
            xtx = torch.matmul(x.transpose(0,1), x) # [h,h]
            if name not in grams:
                grams[name] = xtx / x.size(0)
                xn[name] = x.size(0)
            else:
                grams[name] = (grams[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                xn[name] += x.size(0)
        return hook

    linear_modules = filter_modules_by_regex(model, None, [nn.Linear])
    handles = []
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_gram(name))
        handles.append(handle)

    with torch.no_grad():
        for step, (images, labels) in enumerate(train_dataloader):
            if n_step > 0 and step == n_step:
                break
            images = images.to(device)
            _ = model(images)

    for handle in handles:
        handle.remove()

    return grams

def reduce_non_diag(cov_mat, a):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret

def regmean_merge(p, all_params, all_grams, a=0.9):
    avg_params = {}
    n_model = len(all_grams)
    
    # Identify all parameter names
    param_names = list(all_params.keys())
    
    for name in param_names:
        h_avged = False
        if name.endswith('.weight'):
            module_name = name[:-len('.weight')]
            if len(all_grams) > 0 and module_name in all_grams[0]:
                gram_m_ws, grams = [], []

                for model_id, model_grams in enumerate(all_grams):
                    param_grams = p[model_id] * model_grams[module_name]
                    # Smoothing non-diagonal elements
                    param_grams = reduce_non_diag(param_grams, a=a)

                    param = all_params[name][model_id]
                    # We expect param of shape [out_features, in_features]
                    # RegMean formula: W = (sum G_i)^-1 (sum G_i W_i^T)^T
                    # Or simpler: W^T = (sum G_i)^-1 (sum G_i W_i^T)
                    gram_m_ws.append(torch.matmul(param_grams, param.transpose(0,1)))
                    grams.append(param_grams)
                
                sum_gram = sum(grams)
                sum_gram_m_ws = sum(gram_m_ws)
                # Use pinverse for stability
                sum_gram_inv = torch.pinverse(sum_gram)
                wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)
                w = wt.transpose(0,1)
                avg_params[name] = w
                h_avged = True
        
        if not h_avged: # if not averaged with regmean (e.g. bias, or non-linear layer), then do simple weighted avg
            weighted_params = [p[i] * all_params[name][i] for i in range(len(p))]
            avg_params[name] = torch.stack(weighted_params, 0).sum(0)
           
    return avg_params
