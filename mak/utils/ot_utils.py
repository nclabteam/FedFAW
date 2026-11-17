import copy
import numpy as np
import ot
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from mak.utils.data import evaluate_model
from torch.utils.data import DataLoader

def isnan(x):
    return x != x

class GroundMetric:
    """
    Ground Metric object for Wasserstein computations.
    """
    def __init__(self, params, not_squared=False):
        self.params = params
        self.ground_metric_type = params.get('ground-metric', 'euclidean')
        self.ground_metric_normalize = params.get('ground-metric-normalize', 'none')
        self.reg = params.get('reg', 1e-7)
        self.squared = not params.get('not_squared', False)
        self.mem_eff = params.get('ground-metric-eff', True)

    def _clip(self, ground_metric_matrix):
        # Clipping logic
        clip_min = self.params.get('clip_min', 0)
        clip_max = self.params.get('clip_max', 10)
        ground_metric_matrix.clamp_(min=self.reg * clip_min, max=self.reg * clip_max)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):
        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            ground_metric_matrix = ground_metric_matrix / (ground_metric_matrix.max() + 1e-12)
        elif self.ground_metric_normalize == "median":
            ground_metric_matrix = ground_metric_matrix / (ground_metric_matrix.median() + 1e-12)
        elif self.ground_metric_normalize == "mean":
            ground_metric_matrix = ground_metric_matrix / (ground_metric_matrix.mean() + 1e-12)
        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _pairwise_distances(self, x, y=None, squared=True):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist = torch.clamp(dist, min=0.0)

        if not squared:
            dist = dist ** (1/2)
        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1]) - coordinates, p=2, dim=2
            )
        else:
            matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
        return matrix

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / (torch.norm(coordinates, dim=1, keepdim=True) + 1e-12)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                (torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1) + 1e-12)
            )
        return matrix.clamp_(min=0)

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)
        self._sanity_check(ground_metric_matrix)
        ground_metric_matrix = self._normalize(ground_metric_matrix)
        self._sanity_check(ground_metric_matrix)
        if self.params.get('clip_gm', False):
            ground_metric_matrix = self._clip(ground_metric_matrix)
        return ground_metric_matrix

def get_histogram(args, idx, cardinality):
    # returns a uniform measure as per typical OTFusion baseline without activations
    return np.ones(cardinality) / cardinality

def get_wassersteinized_layers_modularized(args, networks, eps=1e-7):
    model0_aligned_layers = []
    T_var = None
    
    if args['gpu_id'] == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args['gpu_id']))

    ground_metric_object = GroundMetric(args)
    
    # Track skip connections if ResNet18
    skip_T_var = None
    skip_T_var_idx = -1
    residual_T_var = None
    residual_T_var_idx = -1

    for idx, ((layer0_name, weight0), (layer1_name, weight1)) in enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        assert weight0.shape == weight1.shape
        mu_cardinality = weight0.shape[0]
        nu_cardinality = weight1.shape[0]
        layer_shape = weight0.shape

        if len(layer_shape) > 2:
            is_conv = True
            w0_data = weight0.data.view(weight0.shape[0], weight0.shape[1], -1)
            w1_data = weight1.data.view(weight1.shape[0], weight1.shape[1], -1)
        else:
            is_conv = False
            w0_data = weight0.data
            w1_data = weight1.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(w0_data.view(w0_data.shape[0], -1), w1_data.view(w1_data.shape[0], -1))
            else:
                M = ground_metric_object.process(w0_data, w1_data)
            aligned_wt = w0_data
        else:
            if is_conv:
                if args.get('handle_skips', False):
                    # Simplified skip handling logic based on provided code
                    if weight0.shape[1] != weight0.shape[0]:
                        if not (weight0.shape[2] == 1 and weight0.shape[3] == 1):
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx
                            T_var = skip_T_var
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2

                T_var_conv = T_var.unsqueeze(0).repeat(w0_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(w0_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                M = ground_metric_object.process(aligned_wt.contiguous().view(aligned_wt.shape[0], -1), w1_data.view(w1_data.shape[0], -1))
            else:
                if weight0.data.shape[1] != T_var.shape[0]:
                    # Switch from Conv to FC
                    w0_unflattened = weight0.data.view(weight0.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(w0_unflattened, T_var.unsqueeze(0).repeat(w0_unflattened.shape[0], 1, 1)).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(weight0.data, T_var)
                M = ground_metric_object.process(aligned_wt, w1_data)

        mu = get_histogram(args, 0, mu_cardinality)
        nu = get_histogram(args, 1, nu_cardinality)
        cpuM = M.data.cpu().numpy()

        if args.get('exact', True):
            T = ot.emd(mu, nu, cpuM)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.get('reg', 1e-7))

        T_var = torch.from_numpy(T).to(device).float()

        if args.get('correction', False):
            if not args.get('proper_marginals', False):
                marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                marginals = torch.diag(1.0 / (marginals + eps))
                T_var = torch.matmul(T_var, marginals)
            else:
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)
                marginals = (1 / (marginals_beta + eps))
                T_var = T_var * marginals

        if args.get('past_correction', False):
            t_w0 = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_w0 = torch.matmul(T_var.t(), w0_data.view(w0_data.shape[0], -1))

        if is_conv and layer_shape != t_w0.shape:
            t_w0 = t_w0.view(layer_shape)
        model0_aligned_layers.append(t_w0)

    # Reconstruct state dict
    model_aligned = copy.deepcopy(networks[0])
    aligned_state_dict = model_aligned.state_dict()
    for i, key in enumerate(aligned_state_dict.keys()):
        aligned_state_dict[key] = model0_aligned_layers[i]
    model_aligned.load_state_dict(aligned_state_dict)
    return model_aligned

def one_shot_otfusion(net_glob, models, p, args_ot):
    pivot = args_ot.get('pivot', 0)
    num_models = len(models)
    
    # Initialize with pivot model
    pivot_vector = parameters_to_vector(models[pivot].parameters())
    model_avg_vector = p[pivot] * pivot_vector

    for i in range(num_models):
        if i != pivot:
            aligned_model = get_wassersteinized_layers_modularized(args_ot, [models[i], models[pivot]])
            model_avg_vector += p[i] * parameters_to_vector(aligned_model.parameters())

    result_model = copy.deepcopy(net_glob)
    vector_to_parameters(model_avg_vector, result_model.parameters())
    return result_model
