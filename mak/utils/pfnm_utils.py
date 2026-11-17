import copy
import torch
import time
import logging
import numpy as np
from torch import nn
import torch.nn.functional as F
from lapsolver import solve_dense

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SimpleCNNContainerConvBlocks(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, output_dim=10):
        super(SimpleCNNContainerConvBlocks, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class SimpleCNNCIFARContainerConvBlocks(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, output_dim=10):
        super(SimpleCNNCIFARContainerConvBlocks, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, padding=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class LeNetContainer(nn.Module):
    def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(LeNetContainer, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, 1)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class ModerateCNNContainerConvBlocksMNIST(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocksMNIST, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x

class ModerateCNNContainerConvBlocks(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocks, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x

def row_param_cost_simplified(global_weights, weights_j_l, sij_p_gs, red_term):
    match_norms = ((weights_j_l + global_weights) ** 2 / sij_p_gs).sum(axis=1) - red_term
    return match_norms

def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):
    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts, dtype=np.float32), 10)
    sij_p_gs = sigma_inv_j + global_sigmas
    red_term = (global_weights ** 2 / global_sigmas).sum(axis=1)
    param_cost = np.array([row_param_cost_simplified(global_weights, weights_j[l], sij_p_gs, red_term) for l in range(Lj)], dtype=np.float32)
    param_cost += np.log(counts / (J - counts))
    
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added, dtype=np.float32))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)
    full_cost = np.hstack((param_cost, nonparam_cost)).astype(np.float32)
    return full_cost

def matching_upd_j(weights_j, global_weights, sigma_inv_j, global_sigmas, prior_mean_norm, prior_inv_sigma,
                   popularity_counts, gamma, J):
    L = global_weights.shape[0]
    full_cost = compute_cost(global_weights.astype(np.float32), weights_j.astype(np.float32), global_sigmas.astype(np.float32), sigma_inv_j.astype(np.float32), prior_mean_norm.astype(np.float32), prior_inv_sigma.astype(np.float32),
                             popularity_counts, gamma, J)
    row_ind, col_ind = solve_dense(-full_cost)
    assignment_j = []
    new_L = L
    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_weights[i] += weights_j[l]
            global_sigmas[i] += sigma_inv_j
        else:
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_weights = np.vstack((global_weights, prior_mean_norm + weights_j[l]))
            global_sigmas = np.vstack((global_sigmas, prior_inv_sigma + sigma_inv_j))
    return global_weights, global_sigmas, popularity_counts, assignment_j

def patch_weights(w_j, L_next, assignment_j_c):
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    return new_w_j

def block_patching(w_j, L_next, assignment_j_c, layer_index, model_meta_data, 
                                matching_shapes=None, 
                                layer_type="fc", 
                                dataset="cifar10",
                                network_name="lenet"):
    if assignment_j_c is None:
        return w_j
    layer_meta_data = model_meta_data[2 * layer_index - 2]
    if layer_type == "conv":    
        new_w_j = np.zeros((w_j.shape[0], L_next*(layer_meta_data[-1]**2)))
        block_indices = [np.arange(i*layer_meta_data[-1]**2, (i+1)*layer_meta_data[-1]**2) for i in range(L_next)]
        ori_block_indices = [np.arange(i*layer_meta_data[-1]**2, (i+1)*layer_meta_data[-1]**2) for i in range(layer_meta_data[1])]
        for ori_id in range(layer_meta_data[1]):
            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]
    elif layer_type == "fc":
        if network_name == "simple-cnn":
            if dataset in ("cifar10", "cinic10", "svhn", "cifar100"):
                shape_estimator = SimpleCNNContainerConvBlocks(input_channel=3, num_filters=matching_shapes, kernel_size=5)
            elif dataset == "mnist":
                shape_estimator = SimpleCNNContainerConvBlocks(input_channel=1, num_filters=matching_shapes, kernel_size=5)
        elif network_name == "simple-cnn-cifar":
            if dataset in ("cifar10", "cinic10", "svhn", "cifar100"):
                shape_estimator = SimpleCNNCIFARContainerConvBlocks(input_channel=3, num_filters=matching_shapes, kernel_size=5)
            elif dataset == "mnist":
                shape_estimator = SimpleCNNCIFARContainerConvBlocks(input_channel=1, num_filters=matching_shapes, kernel_size=5)
        elif network_name == "moderate-cnn":
            if dataset in ("cifar10", "cinic10"):
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=matching_shapes)
            elif dataset == "mnist":
                shape_estimator = ModerateCNNContainerConvBlocksMNIST(num_filters=matching_shapes)
        elif network_name == "lenet":
            shape_estimator = LeNetContainer(num_filters=matching_shapes, kernel_size=5)
        
        if dataset in ("cifar10", "cinic10", "svhn", "cifar100"):
            dummy_input = torch.rand(1, 3, 32, 32)
        elif dataset == "mnist":
            dummy_input = torch.rand(1, 1, 28, 28)
        
        estimated_output = shape_estimator(dummy_input)
        new_w_j = np.zeros((w_j.shape[0], estimated_output.view(-1).size()[0]))
        prev_layer_meta_data = model_meta_data[2 * layer_index - 2 - 2]
        block_indices = [np.arange(i*estimated_output.size()[-1]**2, (i+1)*estimated_output.size()[-1]**2) for i in range(L_next)]
        ori_block_indices = [np.arange(i*estimated_output.size()[-1]**2, (i+1)*estimated_output.size()[-1]**2) for i in range(prev_layer_meta_data[0])]
        for ori_id in range(prev_layer_meta_data[0]):
            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]
    return new_w_j

def process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0):
    J = len(batch_weights)
    sigma_bias = sigma
    sigma0_bias = sigma0
    mu0_bias = 0.1
    softmax_bias = [batch_weights[j][-1] for j in range(J)]
    softmax_inv_sigma = [s / sigma_bias for s in last_layer_const]
    softmax_bias = sum([b * s for b, s in zip(softmax_bias, softmax_inv_sigma)]) + mu0_bias / sigma0_bias
    softmax_inv_sigma = 1 / sigma0_bias + sum(softmax_inv_sigma)
    return softmax_bias, softmax_inv_sigma

def match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, it):
    J = len(weights_bias)
    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])
    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior
    global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
    global_sigmas = np.outer(np.ones(global_weights.shape[0]), sigma_inv_prior + sigma_inv_layer[group_order[0]])
    popularity_counts = [1] * global_weights.shape[0]
    assignment = [[] for _ in range(J)]
    assignment[group_order[0]] = list(range(global_weights.shape[0]))
    for j in group_order[1:]:
        global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                        global_weights,
                                                                                        sigma_inv_layer[j],
                                                                                        global_sigmas, prior_mean_norm,
                                                                                        sigma_inv_prior,
                                                                                        popularity_counts, gamma, J)
        assignment[j] = assignment_j
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order:
            to_delete = []
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj), assignment[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                else:
                    global_weights[i] = global_weights[i] - batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]
            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)
            global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                            global_weights,
                                                                                            sigma_inv_layer[j],
                                                                                            global_sigmas,
                                                                                            prior_mean_norm,
                                                                                            sigma_inv_prior,
                                                                                            popularity_counts, gamma, J)
            assignment[j] = assignment_j
    return assignment, global_weights, global_sigmas

def layer_wise_group_descent(batch_weights, layer_index, batch_frequencies, sigma_layers, 
                                sigma0_layers, gamma_layers, it, 
                                model_meta_data, 
                                model_layer_type,
                                n_layers,
                                matching_shapes,
                                args):
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]
    last_layer_const = []
    total_freq = sum(batch_frequencies)
    for f in batch_frequencies:
        last_layer_const.append(f / total_freq)
    J = len(batch_weights)
    init_channel_kernel_dims = []
    for bw in batch_weights[0]:
        if len(bw.shape) > 1:
            init_channel_kernel_dims.append(bw.shape[1])
    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    sigma = sigma_layers[layer_index - 1]
    sigma_bias = sigma_bias_layers[layer_index - 1]
    gamma = gamma_layers[layer_index - 1]
    sigma0 = sigma0_layers[layer_index - 1]
    sigma0_bias = sigma0_bias_layers[layer_index - 1]
    if layer_index <= 1:
        weights_bias = [np.hstack((batch_weights[j][0], batch_weights[j][layer_index * 2 - 1].reshape(-1, 1))) for j in range(J)]
        sigma_inv_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias])
        mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias])
        if n_layers == 2:
            sigma_inv_layer = [np.array(D * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
        else:
            sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias]) for j in range(J)]
    elif layer_index == (n_layers - 1) and n_layers > 2:
        weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]
        sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
        mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
        sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in range(J)]
    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        if 'conv' in layer_type or 'features' in layer_type:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2], batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]
        elif 'fc' in layer_type or 'classifier' in layer_type:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]
        sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
        mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
        sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in range(J)]
    assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, it)
    L_next = global_weights_c.shape[0]
    if layer_index <= 1:
        global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[int(layer_index/2)]], global_weights_c[:, init_channel_kernel_dims[int(layer_index/2)]]]
        global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[int(layer_index/2)]], global_sigmas_c[:, init_channel_kernel_dims[int(layer_index/2)]]]
    elif layer_index == (n_layers - 1) and n_layers > 2:
        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape
        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1], global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1], global_sigmas_c[:, gwc_shape[1]-1]]
        elif "fc" in layer_type or 'classifier' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1].T, global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1].T, global_sigmas_c[:, gwc_shape[1]-1]]
    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape
        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1], global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1], global_sigmas_c[:, gwc_shape[1]-1]]
        elif "fc" in layer_type or 'classifier' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1].T, global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1].T, global_sigmas_c[:, gwc_shape[1]-1]]
    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
    return map_out, assignment_c, L_next

class SimpleCNNContainer(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNContainer, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNNCIFARContainer(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNCIFARContainer, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, padding=2)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModerateCNNContainer(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(ModerateCNNContainer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []
    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes
        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]
        freqs.append(np.array(net_freqs))
    return freqs

def pdm_prepare_full_weights_cnn(nets, device="cpu"):
    weights = []
    for net_i, net in enumerate(nets):
        net_weights = []
        statedict = net.state_dict()
        for param_id, (k, v) in enumerate(statedict.items()):
            val = v.cpu().numpy() if device != "cpu" else v.numpy()
            if 'fc' in k or 'classifier' in k:
                if 'weight' in k:
                    net_weights.append(val.T)
                else:
                    net_weights.append(val)
            elif 'conv' in k or 'features' in k:
                if 'weight' in k:
                    if len(v.size()) == 4:
                        net_weights.append(val.reshape(v.size(0), -1))
                else:
                    net_weights.append(val)
        weights.append(net_weights)
    return weights

def local_retrain_dummy(weights, args, freezing_index=0, device="cpu"):
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        matched_cnn = LeNetContainer(num_filters=num_filters, kernel_size=5, input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10)
    elif args.model == "simple-cnn":
        input_channel = 3 if args.dataset in ("cifar10", "cinic10") else 1
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, num_filters=num_filters, kernel_size=5, input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10)
    elif args.model == "simple-cnn-cifar":
        input_channel = 3 if args.dataset in ("cifar10", "cinic10") else 1
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNCIFARContainer(input_channel=input_channel, num_filters=num_filters, kernel_size=5, input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10)
    elif args.model == "moderate-cnn":
        num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
        input_dim = weights[12].shape[0]
        hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
        input_channel = 3 if args.dataset in ("cifar10", "cinic10") else 1
        matched_cnn = ModerateCNNContainer(input_channel, num_filters, kernel_size=3, input_dim=input_dim, hidden_dims=hidden_dims, output_dim=args.n_c)

    new_state_dict = {}
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        val = weights[param_idx]
        if ("fc" in key_name or "classifier" in key_name) and "weight" in key_name:
            val = val.T
        new_state_dict[key_name] = torch.from_numpy(val.reshape(param.size()))
    matched_cnn.load_state_dict(new_state_dict)
    matched_cnn.to(device).train()
    return matched_cnn

def compute_full_cnn(weights, device, args):
    if args.model == "lenet":
        matched_cnn = LeNetContainer(num_filters=[weights[0].shape[0], weights[2].shape[0]], kernel_size=5, input_dim=weights[4].shape[0], hidden_dims=[weights[4].shape[1]], output_dim=10)
    elif args.model == "simple-cnn":
        input_channel = 3 if args.dataset in ("cifar10", "cinic10") else 1
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, num_filters=[weights[0].shape[0], weights[2].shape[0]], kernel_size=5, input_dim=weights[4].shape[0], hidden_dims=[weights[4].shape[1], weights[6].shape[1]], output_dim=10)
    elif args.model == "simple-cnn-cifar":
        input_channel = 3 if args.dataset in ("cifar10", "cinic10") else 1
        matched_cnn = SimpleCNNCIFARContainer(input_channel=input_channel, num_filters=[weights[0].shape[0], weights[2].shape[0]], kernel_size=5, input_dim=weights[4].shape[0], hidden_dims=[weights[4].shape[1], weights[6].shape[1]], output_dim=10)
    elif args.model == "moderate-cnn":
        input_channel = 3 if args.dataset in ("cifar10", "cinic10") else 1
        matched_cnn = ModerateCNNContainer(input_channel, [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]], kernel_size=3, input_dim=weights[12].shape[0], hidden_dims=[weights[12].shape[1], weights[14].shape[1]], output_dim=args.n_c)
    new_state_dict = {}
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        val = weights[param_idx]
        if ("fc" in key_name or "classifier" in key_name) and "weight" in key_name:
            val = val.T
        new_state_dict[key_name] = torch.from_numpy(val.reshape(param.size()))
    matched_cnn.load_state_dict(new_state_dict)
    matched_cnn.to(device)
    return matched_cnn

def oneshot_matching(nets_list, model_meta_data, layer_type, net_dataidx_map, 
                            averaging_weights, args, num_c):
    cls_freqs = net_dataidx_map
    n_classes = num_c
    it=5
    sigma=args.sigma
    sigma0=args.sigma0
    gamma=args.gamma
    assignments_list = []
    batch_weights = pdm_prepare_full_weights_cnn(nets_list, device=args.device)
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(nets_list)
    matching_shapes = []
    first_fc_index = None
    for layer_index in range(1, n_layers):
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
             batch_weights=batch_weights, layer_index=layer_index, sigma0_layers=sigma0, sigma_layers=sigma, 
             batch_frequencies=batch_freqs, it=it, gamma_layers=gamma, model_meta_data=model_meta_data,
             model_layer_type=layer_type, n_layers=n_layers, matching_shapes=matching_shapes, args=args)
        assignments_list.append(assignment)
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2]
        l_type = "conv" if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer else "fc"
        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        if (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and ('conv' in type_of_prev_layer or 'features' in type_of_this_layer)):
            first_fc_index = layer_index
        matching_shapes.append(L_next)
        tempt_weights = [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) for w in range(num_workers)]
        for worker_index in range(num_workers):
            if first_fc_index is None:
                if l_type == "conv":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2], L_next, assignment[worker_index], layer_index+1, model_meta_data, matching_shapes=matching_shapes, layer_type=l_type, dataset=args.dataset, network_name=args.model)
                else:
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, L_next, assignment[worker_index], layer_index+1, model_meta_data, matching_shapes=matching_shapes, layer_type=l_type, dataset=args.dataset, network_name=args.model).T
            else:
                patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, L_next, assignment[worker_index]).T
            tempt_weights[worker_index].append(patched_weight)
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])
        retrained_nets = [local_retrain_dummy(tempt_weights[w], args, freezing_index=(2 * (layer_index + 1) - 2), device=args.device) for w in range(num_workers)]
        batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=args.device)

    matched_weights = []
    last_layer_weights_collector = []
    for i in range(num_workers):
        last_layer_weights_collector.append(np.concatenate((batch_weights[i][-2], batch_weights[i][-1].reshape((1, -1))), axis=0))
    avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)
    for i in range(n_classes):
        avg_weight_collector = sum(averaging_weights[j][i]*last_layer_weights_collector[j][:, i] for j in range(num_workers))
        avg_last_layer_weight[:, i] = avg_weight_collector
    for i in range(len(batch_weights[0]) - 2):
        matched_weights.append(batch_weights[0][i])
    matched_weights.append(avg_last_layer_weight[0:-1, :])
    matched_weights.append(avg_last_layer_weight[-1, :])
    return matched_weights, assignments_list
