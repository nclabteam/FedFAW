import time
import torch
import copy
import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from mak.utils.data import evaluate_model
from mak.utils.compress import quantize_layer
from torch.utils.data import DataLoader

class FedFAW:
    def __init__(self, base_model, models, model_vectors, F_kfac_list, F_diag_list, use_kfac):
        self.base_model = base_model
        self.models = models
        self.model_vectors = model_vectors
        self.F_kfac_list = F_kfac_list
        self.F_diag_list = F_diag_list
        self.use_kfac = use_kfac
        self.device = next(base_model.parameters()).device
        self.d = parameters_to_vector(base_model.parameters()).numel()
        self.optimal_coef = None
        self.history = []
        self.bo_time = None
        self.best_val_acc = None

        # Pre-calculate active Fisher importance vectors
        if self.use_kfac:
            # Derive diagonal importance from KFAC Kronecker factors
            self.active_fisher_list = [
                self._get_kfac_diagonal(f) if f is not None else d 
                for f, d in zip(F_kfac_list, F_diag_list)
            ]
        else:
            # Use raw diagonal importance
            self.active_fisher_list = F_diag_list

    def _get_kfac_diagonal(self, F_kfac):
        """Extract diagonal from KFAC representation."""
        diagonal = []
        for layer_id in F_kfac.data.keys():
            left_factor = F_kfac.data[layer_id][0]
            right_factor = F_kfac.data[layer_id][1]
            diag_left = torch.diag(left_factor).detach()
            diag_right = torch.diag(right_factor).detach()
            layer_diag = torch.kron(diag_left.unsqueeze(1), diag_right.unsqueeze(0))
            diagonal.append(layer_diag.flatten())
        return torch.cat(diagonal).to(self.device).detach()


    def merge(self, dataset_val, args, acquisition_fn='ucb', seed=0, init_points=10, n_iter=50, kappa=2.576, xi=0.0, verbose=True):
        num_clients = len(self.models)
        
        # Define the parameter bounds for Bayesian Optimization
        pbounds = {f'a{i}': (0.0, 1.0) for i in range(num_clients)}

        def black_box_function(**kwargs):
            # 1. Extract and Normalize coefficients (Alpha)
            alphas = [kwargs[f'a{i}'] for i in range(num_clients)]
            sum_alphas = sum(alphas) if sum(alphas) > 0 else 1.0
            normalized_alphas = [a / sum_alphas for a in alphas]

            # 2. Accumulate Fisher-weighted model
            grad_diag_sum = torch.zeros(self.d).to(self.device)
            F_diag_sum = torch.zeros(self.d).to(self.device)
            
            for i in range(num_clients):
                # Apply 15-bit quantization for stability (from stable backup)
                F_i_quant = quantize_layer(self.active_fisher_list[i], self.base_model, 15)
                model_v_quant = quantize_layer(self.model_vectors[i], self.base_model, 15)
                
                grad_diag_sum += F_i_quant * model_v_quant * normalized_alphas[i]
                F_diag_sum += F_i_quant * normalized_alphas[i]
            
            # 3. Handle zero-importance directions with simple average fallback
            ind = torch.where(F_diag_sum < 1e-6)[0]
            model_avg = torch.zeros(self.d).to(self.device)
            for i in range(num_clients):
                model_avg += normalized_alphas[i] * self.model_vectors[i]
                
            fisher_avg = grad_diag_sum / (F_diag_sum + 1e-12)
            fisher_avg[ind] = model_avg[ind]
            
            # 4. Evaluate merged model
            merged_model = copy.deepcopy(self.base_model)
            vector_to_parameters(fisher_avg, merged_model.parameters())
            acc, _ = evaluate_model(merged_model, DataLoader(dataset_val, batch_size=64), self.device)

            # Logging only (does not affect the search): per-evaluation val acc + wall time.
            self.history.append({'eval': len(self.history) + 1, 'val_acc': float(acc),
                                 't': time.perf_counter() - t_start})
            return acc

        # Setup Acquisition Function for bayesian-optimization 2.0.3+
        utility = None
        if acquisition_fn == 'ei':
            utility = acquisition.ExpectedImprovement(xi=xi, random_state=seed)
        elif acquisition_fn == 'poi':
            utility = acquisition.ProbabilityOfImprovement(xi=xi, random_state=seed)
        elif acquisition_fn == 'ucb':
            utility = acquisition.UpperConfidenceBound(kappa=kappa, random_state=seed)

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            acquisition_function=utility,
            random_state=seed,
            verbose=2 if verbose else 0,
        )

        self.history = []
        t_start = time.perf_counter()
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        self.bo_time = time.perf_counter() - t_start

        # Finalize model with best parameters
        best_params = optimizer.max['params']
        self.best_val_acc = float(optimizer.max['target'])
        self.optimal_coef = [best_params[f'a{i}'] for i in range(num_clients)]
        
        # Re-run the best merge for the return model
        return self._do_final_merge(best_params)

    def _do_final_merge(self, params):
        num_clients = len(self.models)
        alphas = [params[f'a{i}'] for i in range(num_clients)]
        sum_alphas = sum(alphas) if sum(alphas) > 0 else 1.0
        normalized_alphas = [a / sum_alphas for a in alphas]

        grad_diag_sum = torch.zeros(self.d).to(self.device)
        F_diag_sum = torch.zeros(self.d).to(self.device)
        
        for i in range(num_clients):
            F_i_quant = quantize_layer(self.active_fisher_list[i], self.base_model, 15)
            model_v_quant = quantize_layer(self.model_vectors[i], self.base_model, 15)
            grad_diag_sum += F_i_quant * model_v_quant * normalized_alphas[i]
            F_diag_sum += F_i_quant * normalized_alphas[i]
        
        ind = torch.where(F_diag_sum < 1e-6)[0]
        model_avg = torch.zeros(self.d).to(self.device)
        for i in range(num_clients):
            model_avg += normalized_alphas[i] * self.model_vectors[i]
            
        fisher_avg = grad_diag_sum / (F_diag_sum + 1e-12)
        fisher_avg[ind] = model_avg[ind]
        
        final_model = copy.deepcopy(self.base_model)
        vector_to_parameters(fisher_avg, final_model.parameters())
        return final_model

    def get_optimal_coefficients(self):
        return self.optimal_coef
