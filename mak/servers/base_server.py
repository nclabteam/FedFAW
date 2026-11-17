import os
import torch
from tqdm import tqdm
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mak.utils.data import dataset_info
from mak.utils.comm_utils import get_size_mb





class FedAvgServer:
    def __init__(self, model, config, logger, device, out_dir, validation_dataset):
        self.config = config
        self.total_clients = config['server']['num_clients']
        self.fraction_fit = config['server']['fraction_fit']
        self.num_join_clients = int(self.total_clients * self.fraction_fit)
        self.global_model = model
        self.logger = logger
        self.device = device
        self.out_dir = out_dir
        self.validation_dataset = validation_dataset

        self.verbose = config['common']['verbose']

        self.dataset_cfg = dataset_info[self.config['common']['dataset']]
        self.channel = self.dataset_cfg.get('channel')
        self.num_classes = self.dataset_cfg['num_classes']
        self.im_size = self.dataset_cfg.get('img_size')
        self.mean = self.dataset_cfg.get('mean')
        self.std = self.dataset_cfg.get('std')
        self.logger.info("Server intilized : FedAvg Server")
        self.logger.info("Server initialized with global model")

    def __repr__(self) -> str:
        return " FedAvg Strategy"

    def get_global_model(self):
        return self.global_model.state_dict()

    def fit_round(self, clients, round):
        results, upload_cost, download_cost = self.fit(
            clients=clients, round=round)
        self.logger.info(
            f"Training Round : {round} complete Aggregating results")
        self.aggregate(results)
        return upload_cost, download_cost

    def fit(self, clients, round):
        results = {}
        upload_cost = 0
        download_cost = 0
        global_model_size = get_size_mb(self.get_global_model())
        for c_id in tqdm(clients.keys(), desc=f'Round : {round} Training Selected Clients'):
            # logger.info(f"Training client {c_id} in round : {round}")
            # Download cost: server -> client
            download_cost += global_model_size

            client = clients[c_id]
            client.set_model(copy.deepcopy(self.get_global_model()))
            trained_model = client.train()

            # Upload cost: client -> server
            upload_cost += get_size_mb(trained_model)

            results[c_id] = {'model': trained_model,
                             'num_samples': client.data_size}
        return results, upload_cost, download_cost

    def aggregate(self, results):
        """Aggregates client models using FedAvg with weighting by data sizes."""
        self.logger.info("Aggregating client models with data size weighting.")
        global_dict = self.global_model.state_dict()
        # Calculate total_samples
        total_samples = sum(entry['num_samples'] for entry in results.values())

        # Sum the weighted parameters from each client
        # Initialize averaged_params
        averaged_params = {key: 0 for key in global_dict.keys()}

        for client_data in results.values():
            client_model = client_data['model']  # Access the trained model
            # Access the number of samples
            data_size = client_data['num_samples']

            # Calculate weight
            weight = data_size / total_samples

            # Update averaged_params with weighted parameters
            for key in global_dict.keys():
                averaged_params[key] += client_model[key] * weight

        # Update the global model parameters
        self.global_model.load_state_dict(averaged_params)
        self.logger.info("Model aggregation completed.")



    def evaluate(self, test_loader, round_num):
        """Evaluates the global model on test data and saves the model."""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                test_loss += F.cross_entropy(outputs, labels,
                                             reduction='sum').item()
                y_pred = outputs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)
                                     ).long().cpu().sum()
                total += labels.size(0)

        average_loss = test_loss / total
        accuracy = 100.00 * correct / total
        if isinstance(accuracy, torch.Tensor):
            accuracy = accuracy.item()

        # # Save the global model after evaluation
        model_save_path = f"{self.out_dir}/models/fedavg_global_model_{round_num}.pth"

        # # Extract the directory path from the model save path
        model_save_dir = os.path.dirname(model_save_path)

        # Check if the directory exists, if not, create it
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # torch.save(self.global_model.state_dict(), model_save_path)

        self.logger.info(
            "Global Model Accuracy: %.2f%%, Loss: %.4f", accuracy, average_loss)
        # self.logger.info("Global model saved at: %s", model_save_path)

        return accuracy, average_loss
