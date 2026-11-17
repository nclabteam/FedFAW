import os
import torch
from tqdm import tqdm
import copy
from mak.utils.data import dataset_info
from mak.servers.base_server import FedAvgServer
from mak.utils.comm_utils import get_size_mb


class ScaffoldServer(FedAvgServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset):
        super().__init__(model, config, logger, device, out_dir,validation_dataset)
        self.server_cv = [torch.zeros_like(
            param, device=self.device) for param in self.global_model.parameters()]
        self.server_learning_rate = 1.0

    def __repr__(self) -> str:
        return " Scaffold Strategy"

    def get_global_model(self):
        return self.global_model.state_dict()

    def fit_round(self, clients, round):
        results, upload_cost, download_cost = self.fit(
            clients=clients, round=round)
        self.logger.info(
            f"Training Round : {round} complete Aggregating results")
        self.aggregate(results)
        return upload_cost, download_cost

        # scaffold update server c_v and model

    def fit(self, clients, round):
        results = {}
        upload_cost = 0
        download_cost = 0
        global_model_size = get_size_mb(self.get_global_model())
        server_cv_size = get_size_mb(self.server_cv)

        for c_id in tqdm(clients.keys(), desc=f'Round : {round} Training Selected Clients'):
            # Download cost: model + server_cv
            download_cost += (global_model_size + server_cv_size)

            client = clients[c_id]
            # set client.server_cv as well here
            client.server_cv = copy.deepcopy(self.server_cv)
            client.set_model(copy.deepcopy(self.get_global_model()))
            trained_model = client.train()
            delta_y, delta_c = client.get_delta_yc()

            # Upload cost: model + delta_y + delta_c
            # (Note: trained_model is the updated local model)
            upload_cost += (get_size_mb(trained_model) +
                            get_size_mb(delta_y) + get_size_mb(delta_c))

            results[c_id] = {'model': trained_model, 'num_samples': client.data_size,
                             'delta_y': delta_y, 'delta_c': delta_c}
        return results, upload_cost, download_cost

    def aggregate(self, results):
        """Aggregates client models using FedAvg with weighting by data sizes."""
        self.logger.info("Aggregating client models and control variates .")
        global_model = copy.deepcopy(self.global_model)
        server_cv = copy.deepcopy(self.server_cv)

        for client_data in results.values():
            client_model = client_data['model']  # Access the trained model
            # Access the number of samples
            data_size = client_data['num_samples']
            dy = client_data['delta_y']
            dc = client_data['delta_c']

            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / self.num_join_clients * \
                    self.server_learning_rate
            for server_param, client_param in zip(server_cv, dc):
                server_param.data += client_param.data.clone() / self.total_clients
        self.global_model = global_model
        self.server_cv = server_cv
        self.logger.info("Model aggregation completed.")
