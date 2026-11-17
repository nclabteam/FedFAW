import os
import torch
from tqdm import tqdm
import copy
from mak.utils.data import dataset_info, evaluate_model
from mak.utils.comm_utils import get_size_mb

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from mak.servers.base_server import FedAvgServer
from torch.utils.data import DataLoader
from mak.utils.compress import quantize_layer, compress_fisher_kfac, get_dot_product, element_mul, inv_sqrt
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector


class FedFisherServer(FedAvgServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset, full_dataset, client_indices):
        super().__init__(model, config, logger, device, out_dir, validation_dataset)
        self.config = config
        self.total_clients = config['server']['num_clients']
        self.fraction_fit = config['server']['fraction_fit']
        self.num_join_clients = int(self.total_clients * self.fraction_fit)
        self.global_model = model
        self.logger = logger
        self.device = device
        self.out_dir = out_dir

        self.verbose = config['common']['verbose']

        self.dataset_cfg = dataset_info[self.config['common']['dataset']]
        self.channel = self.dataset_cfg.get('channel')
        self.num_classes = self.dataset_cfg['num_classes']
        self.im_size = self.dataset_cfg.get('img_size')
        self.mean = self.dataset_cfg.get('mean')
        self.std = self.dataset_cfg.get('std')
        self.d = parameters_to_vector(self.global_model.parameters()).numel()
        self.validation_dataset = DataLoader(
            validation_dataset, batch_size=self.config['client']['test_batch_size'], shuffle=False)
        self.full_dataset = full_dataset
        self.client_indices = client_indices

        self.fisher_type = self.config['fisher']['fisher_type']

        self.logger.info("Server initialized : FedFisher Server")
        self.logger.info("Server initialized with global model")

    def __repr__(self) -> str:
        return " Fisher Strategy"

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
            trained_model, fisher_matrix = client.train()

            # Upload cost: client -> server (Model + Fisher Matrix)
            upload_cost += get_size_mb(trained_model)
            upload_cost += get_size_mb(fisher_matrix)

            results[c_id] = {'model': trained_model,
                             'num_samples': client.data_size,
                             'fisher_matrix': fisher_matrix}
        return results, upload_cost, download_cost

    def aggregate(self, results):
        client_weights = {}
        total_samples = sum(entry['num_samples'] for entry in results.values())
        model_vectors = []
        fisher_vectors = []
        models = []

        # We use a copy of the global model to load state_dicts into
        sample_model = copy.deepcopy(self.global_model)

        weights_list = []
        for c_id, client_data in results.items():
            # Access the trained model (state_dict)
            client_model = client_data['model']
            data_size = client_data['num_samples']
            fisher_matrix = client_data['fisher_matrix']

            # Load the client's state_dict into our sample model
            # We must use a separate copy if we want to store it in a list
            client_model_obj = copy.deepcopy(self.global_model)
            client_model_obj.load_state_dict(client_model)
            models.append(client_model_obj)

            # Convert parameters to vector
            model_vectors.append(
                parameters_to_vector(client_model_obj.parameters()))

            # Calculate and store weight
            weight = data_size / total_samples
            client_weights[c_id] = weight
            weights_list.append(weight)

            # Weighted Fisher matrix
            if self.fisher_type == 'diagonal':
                fisher_vectors.append(fisher_matrix * weight)
            else:
                for layer_id in fisher_matrix.data.keys():
                    fisher_matrix.data[layer_id] = list(
                        fisher_matrix.data[layer_id])
                    fisher_matrix.data[layer_id][0].mul_(weight)
                    fisher_matrix.data[layer_id][1].mul_(weight)
                fisher_vectors.append(fisher_matrix)

        updated_model = self.aggregate_model(
            self.d, weights_list, models, model_vectors, fisher_vectors)
        # Update the global model parameters
        self.global_model.load_state_dict(updated_model.state_dict())
        self.logger.info("Model aggregation completed.")

    def aggregate_model(self, d, client_weights, models, model_vectors, fisher_vectors):
        num_clients = len(model_vectors)

        if self.config['fisher']['fisher_type'] == 'diagonal':
            fisher_avg_model = self.fisher_diagonal(
                num_clients, model_vectors, fisher_vectors, client_weights)
        elif self.config['fisher']['fisher_type'] == 'kfac':
            fisher_avg_model = self.fisher_kfac(
                num_clients, models, model_vectors, fisher_vectors, client_weights, self.validation_dataset)
        else:
            raise ValueError("Invalid fisher type")

        return fisher_avg_model

    def fisher_diagonal(self, num_clients, model_vectors, fisher_vectors, client_weights):
        grad_diag_sum = torch.zeros(self.d).to(self.device)
        F_diag_sum = torch.zeros(self.d).to(self.device)
        model_vector_sum = torch.zeros(self.d).to(self.device)

        for i in range(num_clients):
            F_diag_quantized = quantize_layer(
                fisher_vectors[i], self.global_model, 15)
            model_vector_quantized = quantize_layer(
                model_vectors[i], self.global_model, 15)
            grad_diag = F_diag_quantized*model_vector_quantized
            grad_diag_sum += grad_diag
            F_diag_sum += F_diag_quantized
            model_vector_sum += client_weights[i]*model_vector_quantized

        fisher_avg_model = copy.deepcopy(self.global_model)
        fedavg_quantized_model = copy.deepcopy(self.global_model)
        vector_to_parameters(
            model_vector_sum, fedavg_quantized_model.parameters())
        fisher_avg = self.one_shot_fisher_diag(fedavg_quantized_model, F_diag_sum, grad_diag_sum,
                                               client_weights, self.validation_dataset, self.config['fisher'], self.config)
        vector_to_parameters(fisher_avg, fisher_avg_model.parameters())
        return fisher_avg_model

    def fisher_kfac(self, num_clients, models, model_vectors, fisher_vectors, client_weights, validation_dataset):
        model_vector_sum = torch.zeros(self.d).to(self.device)
        for i in range(num_clients):
            model_vector_quantized = quantize_layer(
                model_vectors[i], self.global_model, 15)
            model_vector_sum += client_weights[i]*model_vector_quantized

        fedavg_quantized_model = copy.deepcopy(self.global_model)
        vector_to_parameters(
            model_vector_sum, fedavg_quantized_model.parameters())
        F_kfac_list_comp, grad_avg_comp = compress_fisher_kfac(
            self.config, fisher_vectors, models, model_vectors, self.full_dataset, client_weights, self.num_classes, self.client_indices, self.device)
        fisher_avg_model = copy.deepcopy(self.global_model)
        fisher_avg = self.one_shot_fisher_kfac(fedavg_quantized_model, F_kfac_list_comp,
                                               grad_avg_comp, client_weights, validation_dataset, self.config['fisher'], self.config)
        vector_to_parameters(fisher_avg, fisher_avg_model.parameters())
        return fisher_avg_model

    def one_shot_fisher_diag(self, model, F_diag, grad_diag, p, dataset_val, args_fisher, args):
        net_glob_copy = copy.deepcopy(model)
        w_avg = parameters_to_vector(net_glob_copy.parameters())
        w = parameters_to_vector(net_glob_copy.parameters())
        T = args_fisher['server_iterations']
        eta = args_fisher['server_lr']
        test_acc_i_max = 0

        with torch.no_grad():

            test_acc_tracker = []
            test_acc_i_max = 0
            best_parameters = None
            mom = 0*w
            delta = 0*w

            # Using Adam optimizer for the global optimization in FedFisher(Diag)
            for k in range(T):
                v = F_diag*w - grad_diag
                mom = v + mom*0.9
                delta = v*v + delta*0.99
                w = w - eta*mom*1/(torch.sqrt(delta) + 0.01)

                if (k % 100 == 0):

                    w_vec_estimate = w
                    vector_to_parameters(
                        w_vec_estimate, net_glob_copy.parameters())
                    test_acc_i, test_loss_i = evaluate_model(
                        net_glob_copy, self.validation_dataset, self.device)
                    if (test_acc_i > test_acc_i_max):
                        test_acc_i_max = test_acc_i
                        best_parameters = w

                    print("Val Test Acc: ", test_acc_i,
                          " Val Test Loss: ", test_loss_i)
                    test_acc_tracker.append(test_acc_i)

        return best_parameters

    def one_shot_fisher_kfac(self, model, F_kfac_list_comp, grad_avg_comp, p, dataset_val, args_fisher, args):
        w = PVector.from_model(model)
        w_avg = PVector.from_model(model)
        net_glob_copy = copy.deepcopy(model)
        T = args_fisher['server_iterations']
        eta = args_fisher['server_lr']
        test_acc_i_max = 0
        test_acc_tracker = []

        with torch.no_grad():

            best_parameters = copy.deepcopy(w.get_flat_representation())
            mom = w.__rmul__(0)
            delta = w.__rmul__(0)

            # Using Adam optimizer for the global optimization in FedFisher(K-FAC)
            for k in range(T):

                v = get_dot_product(F_kfac_list_comp, w, p)
                v = v.__sub__(grad_avg_comp)
                mom = v.__add__(mom.__rmul__(0.9))
                delta = element_mul(v, v).__add__(delta.__rmul__(0.99))
                v = element_mul(mom, inv_sqrt(delta, 0.01))
                v = v.__rmul__(eta)
                w = w.__sub__(v)

                if (k % 100 == 0):
                    w_vec_estimate = w.get_flat_representation()
                    vector_to_parameters(
                        w_vec_estimate, net_glob_copy.parameters())
                    test_acc_i, test_loss_i = evaluate_model(
                        net_glob_copy, self.validation_dataset, self.device)
                    if (test_acc_i > test_acc_i_max):
                        test_acc_i_max = test_acc_i
                        best_parameters = w_vec_estimate

                    print("Val Test Acc: ", test_acc_i,
                          " Val Test Loss: ", test_loss_i)
                    test_acc_tracker.append(test_acc_i)

        return best_parameters
