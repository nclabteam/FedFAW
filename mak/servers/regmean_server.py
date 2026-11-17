import copy
import torch
from tqdm import tqdm
from mak.servers.base_server import FedAvgServer
from mak.utils.data import evaluate_model
from mak.utils.regmean_utils import compute_gram, regmean_merge
from mak.utils.compress import compress_regmean_gram, quantize_layer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset

class RegMeanServer(FedAvgServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset, full_dataset, client_indices):
        super().__init__(model, config, logger, device, out_dir, validation_dataset)
        self.validation_dataset = DataLoader(
            validation_dataset, batch_size=self.config['client']['test_batch_size'], shuffle=False)
        self.full_dataset = full_dataset
        self.client_indices = client_indices
        # RegMean requires every client to send per-layer input Gram
        # matrices to the server on top of the model weights. We compute
        # them server-side in aggregate() as a simulation shortcut (the
        # server doesn't have client data in a real deployment), so this
        # tracks their transmission cost -- populated each round by
        # aggregate() and added to upload_cost in fit_round() -- so the
        # reported communication cost isn't silently identical to FedAvg's.
        self.last_gram_upload_mb = 0.0
        self.logger.info("Server initialized: RegMean Server")

    def fit_round(self, clients, round):
        results, upload_cost, download_cost = self.fit(clients=clients, round=round)
        self.logger.info(f"Training Round : {round} complete Aggregating results")
        self.aggregate(results)
        upload_cost += self.last_gram_upload_mb
        return upload_cost, download_cost


    def aggregate(self, results):
        total_samples = sum(entry['num_samples'] for entry in results.values())
        model_vectors = []
        models = []
        p = []
        
        # Sort results for consistency
        sorted_keys = sorted(results.keys())
        for c_id in sorted_keys:
            client_data = results[c_id]
            # Convert state dict to model object
            client_model = copy.deepcopy(self.global_model)
            client_model.load_state_dict(client_data['model'])
            
            # Use quantized version if needed as per user snippet
            model_vector = parameters_to_vector(client_model.parameters())
            model_vector_quantized = quantize_layer(model_vector, client_model, 15)
            vector_to_parameters(model_vector_quantized, client_model.parameters())
            
            models.append(client_model)
            model_vectors.append(model_vector_quantized)
            p.append(client_data['num_samples'] / total_samples)

        self.logger.info("Computing Gram matrices for RegMean...")
        grams_list = []
        for i, c_id in enumerate(sorted_keys):
            # In simulation, we use the server's access to local datasets
            client_dataset = Subset(self.full_dataset, self.client_indices[c_id])
            grams = compute_gram(models[i], client_dataset, device=self.device)
            grams_list.append(grams)

        # Optional compression -- also returns the compressed Gram-matrix
        # payload size (bits, summed across all clients), i.e. the extra
        # data a real deployment would need clients to upload beyond the
        # model weights.
        gram_bits = compress_regmean_gram(grams_list, self.global_model)
        self.last_gram_upload_mb = gram_bits / 8 / (1024 * 1024)

        self.logger.info("Performing RegMean aggregation with hyperparameter tuning...")
        
        updated_model_params = self.get_regmean_avg_params(p, models, grams_list)
        
        # Update the global model parameters
        self.global_model.load_state_dict(updated_model_params)
        self.logger.info("RegMean aggregation completed.")

    def get_regmean_avg_params(self, p, models, grams_list):
        # Prepare params dictionary for regmean_merge
        all_params = {}
        for model in models:
            for name, param in model.named_parameters():
                if name not in all_params:
                    all_params[name] = []
                all_params[name].append(param.data)

        # Range for 'a' parameter (non-diagonal reduction)
        a_range = self.config.get('regmean', {}).get('a_range', [0.1, 0.9])
        best_params = None
        best_acc = -1.0

        for a in a_range:
            regmean_avg_params = regmean_merge(p, all_params, grams_list, a=a)
            
            # Evaluate candidate model
            candidate_model = copy.deepcopy(self.global_model)
            candidate_model.load_state_dict(regmean_avg_params)
            acc, _ = evaluate_model(candidate_model, self.validation_dataset, self.device)
            
            self.logger.info(f"RegMean Params - a: {a} => Val Acc: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_params = regmean_avg_params

        return best_params

    def __repr__(self) -> str:
        return " RegMean Strategy"
