import copy
import torch
from mak.servers.fedfisher_server import FedFisherServer
from mak.utils.fedlpa_utils import newton_solve
from collections import OrderedDict
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class FedLPAServer(FedFisherServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset, full_dataset, client_indices):
        super().__init__(model, config, logger, device, out_dir, validation_dataset, full_dataset, client_indices)
        self.logger.info("Server initialized: FedLPA Server")
        
        self.global_fim = OrderedDict()
        for name, module in self.global_model.named_modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                self.global_fim[name] = [None, None]

    def fit_round(self, clients, round):
        # Broadcast global model and global FIM to clients
        mu_global = self.global_model.state_dict()
        f_global = self.global_fim
        
        for client_id in clients:
            if hasattr(clients[client_id], 'set_global_info'):
                clients[client_id].set_global_info(mu_global, f_global)
            
        return super().fit_round(clients, round)

    def aggregate(self, results):
        n = len(results)
        if n == 0: return
        
        total_samples = sum(results[c_id]['num_samples'] for c_id in results)
        sorted_keys = sorted(results.keys())
        fed_avg_freqs = [results[c_id]['num_samples'] / total_samples for c_id in sorted_keys]
        
        # 1. Standard FedAvg weighted parameters (base for all layers)
        avg_param = OrderedDict()
        for idx, c_id in enumerate(sorted_keys):
            net_para = results[c_id]['model']
            freq = fed_avg_freqs[idx]
            for key in net_para:
                if idx == 0:
                    avg_param[key] = freq * net_para[key].to(self.device).clone()
                else:
                    avg_param[key] += freq * net_para[key].to(self.device)

        # 2. Accumulate Z and Global FIM for Linear/Conv layers
        global_temp_para = {}
        for idx, c_id in enumerate(sorted_keys):
            local_model_state = results[c_id]['model']
            local_fim = results[c_id]['fisher_matrix'] 
            freq = fed_avg_freqs[idx]
            
            for name in self.global_fim:
                # Prepare local M_k = [W_k, b_k]
                W_k_orig = local_model_state[name + '.weight']
                weight_shape = W_k_orig.shape
                
                if len(weight_shape) == 4:
                    W_k = W_k_orig.reshape(weight_shape[0], -1)
                else:
                    W_k = W_k_orig
                
                if name + '.bias' in local_model_state:
                    M_k = torch.cat([W_k.to(self.device), local_model_state[name + '.bias'].unsqueeze(1).to(self.device)], dim=1)
                else:
                    M_k = W_k.to(self.device)
                
                # Biased target contribution: Z_k = B_k @ M_k @ Ak
                Ak = local_fim[name][0].to(self.device)
                Bk = local_fim[name][1].to(self.device)
                Z_k = Bk @ M_k @ Ak
                
                if idx == 0:
                    self.global_fim[name][0] = freq * Ak
                    self.global_fim[name][1] = freq * Bk
                    global_temp_para[name] = freq * Z_k
                else:
                    self.global_fim[name][0] += freq * Ak
                    self.global_fim[name][1] += freq * Bk
                    global_temp_para[name] += freq * Z_k

        # 3. Solve for New Global Parameters (Newton Approximation)
        newtons_param = copy.deepcopy(avg_param)
        biased_param = copy.deepcopy(avg_param) # For logging reference if needed
        
        self.logger.info("FedLPA: Solving global alignment via Newton's method...")
        for name in self.global_fim:
            Ak_list = [results[c_id]['fisher_matrix'][name][0].to(self.device) for c_id in sorted_keys]
            Bk_list = [results[c_id]['fisher_matrix'][name][1].to(self.device) for c_id in sorted_keys]
            
            # Newton Solve
            M_newton = newton_solve(fed_avg_freqs, Ak_list, Bk_list, global_temp_para[name], self.device, iterations=1000)
            
            # Biased solution (Expectation Approximation)
            eps = 1e-10
            G_inv = torch.inverse(self.global_fim[name][1] + eps * torch.eye(self.global_fim[name][1].size(0)).to(self.device))
            A_inv = torch.inverse(self.global_fim[name][0] + eps * torch.eye(self.global_fim[name][0].size(0)).to(self.device))
            M_biased = G_inv @ global_temp_para[name] @ A_inv


            # Unpack M_newton to state_dict
            weight_orig_shape = avg_param[name + '.weight'].shape
            if name + '.bias' in avg_param:
                newtons_param[name + '.weight'] = M_newton[:, :-1].reshape(weight_orig_shape)
                newtons_param[name + '.bias'] = M_newton[:, -1]
                
                biased_param[name + '.weight'] = M_biased[:, :-1].reshape(weight_orig_shape)
                biased_param[name + '.bias'] = M_biased[:, -1]
            else:
                newtons_param[name + '.weight'] = M_newton.reshape(weight_orig_shape)
                biased_param[name + '.weight'] = M_biased.reshape(weight_orig_shape)

        # Update global model with Newton's results
        self.global_model.load_state_dict(newtons_param)
        self.logger.info("FedLPA aggregation completed.")

    def __repr__(self) -> str:
        return " FedLPA Strategy"
