import os
import torch
from tqdm import tqdm
import copy
from mak.utils.data import dataset_info, evaluate_model
from mak.utils.comm_utils import get_size_mb

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from mak.servers.fedfisher_server import FedFisherServer
from torch.utils.data import DataLoader
from mak.utils.compress import quantize_layer

class FisherMergeServer(FedFisherServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset, full_dataset, client_indices):
        super().__init__(model, config, logger, device, out_dir, validation_dataset, full_dataset, client_indices)
        self.logger.info("Server initialized : Fisher Merge Server")
        self.fisher_type = 'diagonal'

    def __repr__(self) -> str:
        return " Fisher Merge Strategy"

    def aggregate(self, results):
        """
        Implementation of the analytical Fisher Merge logic.
        """
        self.logger.info("+++++++++++++ Aggregating models using Fisher Merge strategy +++++++++++++")
        
        client_weights = []
        model_vectors = []
        fisher_vectors = []
        total_samples = sum(entry['num_samples'] for entry in results.values())
        sorted_keys = sorted(results.keys())

        for c_id in sorted_keys:
            client_data = results[c_id]
            client_model = client_data['model']
            data_size = client_data['num_samples']
            fisher_matrix = client_data['fisher_matrix']
            
            client_model_obj = copy.deepcopy(self.global_model)
            client_model_obj.load_state_dict(client_model)
            
            model_vectors.append(parameters_to_vector(client_model_obj.parameters()))
            
            weight = data_size / total_samples
            client_weights.append(weight)
            fisher_vectors.append(fisher_matrix * weight)

        num_clients = len(model_vectors)
        grad_diag_sum = torch.zeros(self.d).to(self.device)
        F_diag_sum = torch.zeros(self.d).to(self.device)
        model_vector_sum = torch.zeros(self.d).to(self.device)
        model_avg = torch.zeros(self.d).to(self.device)

        for i in range(num_clients):
            model_avg += client_weights[i] * model_vectors[i]

        for i in range(num_clients):
            F_diag_quantized = quantize_layer(fisher_vectors[i], self.global_model, 15)
            model_vector_quantized = quantize_layer(model_vectors[i], self.global_model, 15)
            
            grad_diag = F_diag_quantized * model_vector_quantized
            grad_diag_sum += grad_diag
            F_diag_sum += F_diag_quantized
            model_vector_sum += client_weights[i] * model_vector_quantized

        ind = torch.where(F_diag_sum < 1e-6)[0].detach().cpu().numpy()
        
        fisher_avg_model = copy.deepcopy(self.global_model)
        fedavg_quantized_model = copy.deepcopy(self.global_model)
        vector_to_parameters(model_vector_sum, fedavg_quantized_model.parameters())
        
        fisher_avg = grad_diag_sum / F_diag_sum
        fisher_avg[ind] = model_avg[ind]
        
        vector_to_parameters(fisher_avg, fisher_avg_model.parameters())
        
        # Update current global model
        self.global_model.load_state_dict(fisher_avg_model.state_dict())
        self.logger.info("Model aggregation completed.")