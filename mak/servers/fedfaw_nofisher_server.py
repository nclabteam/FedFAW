import copy
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from bayes_opt import BayesianOptimization, acquisition
from torch.utils.data import DataLoader

from mak.servers.base_server import FedAvgServer
from mak.utils.data import dataset_info, evaluate_model
from mak.utils.comm_utils import get_size_mb


class FedFAWNoFisherServer(FedAvgServer):
    """
    Ablation arm: BO-optimized client weights + plain weighted average (no Fisher).

    This isolates the contribution of validation-guided client weighting
    from the Fisher aggregation mechanism. Compared to the full FedFAW:
      - FedFAW    : Fisher-weighted agg  + BO client weights  (full method)
      - FedFAW w/o Fisher: Plain weighted avg   + BO client weights  (this class)
      - FedFisher  : Fisher-weighted agg  + static (p_i) weights
      - FedAvg     : Plain weighted avg   + static (p_i) weights

    The aggregation rule for a given alpha vector is:
        theta_agg(alpha) = sum_i alpha_i * theta_i
    where alpha is selected by maximising validation accuracy via BO.
    """

    def __init__(self, model, config, logger, device, out_dir,
                 validation_dataset, full_dataset, client_indices):
        super().__init__(model=model, config=config, logger=logger,
                         device=device, out_dir=out_dir,
                         validation_dataset=validation_dataset)
        self.config = config
        self.global_model = model
        self.logger = logger
        self.device = device
        self.out_dir = out_dir
        self.verbose = config['common']['verbose']

        dataset_name = config['common']['dataset']
        self.d = parameters_to_vector(self.global_model.parameters()).numel()

        # Wrap validation dataset in a DataLoader for evaluate_model()
        self.validation_dataset = DataLoader(
            validation_dataset,
            batch_size=config['client']['test_batch_size'],
            shuffle=False
        )

        # BO hyperparameters — shared with fedfaw block for consistency
        faw_cfg = config.get('fedfaw', {})
        self.init_points = faw_cfg.get('init_points', 10)
        self.n_iter = faw_cfg.get('n_iter', 50)
        self.acq = faw_cfg.get('acquisition_fn', 'ucb')
        self.kappa = faw_cfg.get('kappa', 2.576)
        self.xi = faw_cfg.get('xi', 0.01)

        self.logger.info(
            f"Server initialized: FedFAW w/o Fisher Server "
            f"(BO budget: init={self.init_points}, iter={self.n_iter}, "
            f"acq={self.acq})"
        )

    def __repr__(self) -> str:
        return "FedFAW w/o Fisher Strategy (plain avg + BO weights, no Fisher)"

    # ------------------------------------------------------------------ #
    #  fit / fit_round — identical pattern to FedFisherServer              #
    # ------------------------------------------------------------------ #

    def fit_round(self, clients, round):
        results, upload_cost, download_cost = self.fit(
            clients=clients, round=round)
        self.logger.info(
            f"Training Round : {round} complete. Aggregating results.")
        self.aggregate(results)
        return upload_cost, download_cost

    def fit(self, clients, round):
        results = {}
        upload_cost = 0
        download_cost = 0
        global_model_size = get_size_mb(self.get_global_model())

        for c_id in tqdm(clients.keys(),
                         desc=f'Round {round} Training Selected Clients'):
            download_cost += global_model_size

            client = clients[c_id]
            client.set_model(copy.deepcopy(self.get_global_model()))
            # BaseClient.train() returns only the state_dict (no Fisher)
            trained_model = client.train()

            upload_cost += get_size_mb(trained_model)

            results[c_id] = {
                'model': trained_model,
                'num_samples': client.data_size,
            }

        return results, upload_cost, download_cost

    # ------------------------------------------------------------------ #
    #  aggregate                                                           #
    # ------------------------------------------------------------------ #

    def aggregate(self, results):
        total_samples = sum(e['num_samples'] for e in results.values())
        model_vectors = []
        static_weights = []   # p_i = n_i / N  (kept for reference / logging)

        sorted_keys = sorted(results.keys())
        for c_id in sorted_keys:
            client_data = results[c_id]
            client_model_obj = copy.deepcopy(self.global_model)
            client_model_obj.load_state_dict(client_data['model'])

            model_vectors.append(
                parameters_to_vector(client_model_obj.parameters()).detach()
            )
            static_weights.append(client_data['num_samples'] / total_samples)

        self.logger.info(
            f"Static (sample-size) weights for reference: {static_weights}")

        # Run BO over plain averaging weights
        updated_model = self._bo_plain_average(model_vectors)
        self.global_model.load_state_dict(updated_model.state_dict())
        self.logger.info("FedFAW w/o Fisher model aggregation completed.")

    # ------------------------------------------------------------------ #
    #  BO over plain weighted average                                      #
    # ------------------------------------------------------------------ #

    def _bo_plain_average(self, model_vectors):
        """
        Runs Bayesian optimisation to find alpha* that maximises validation
        accuracy when models are combined as theta_agg = sum_i alpha_i theta_i.
        No Fisher information is used anywhere.
        """
        num_clients = len(model_vectors)
        d = self.d
        device = self.device

        pbounds = {f'a{i}': (0.0, 1.0) for i in range(num_clients)}

        def black_box_function(**kwargs):
            # 1. Extract and l1-normalise to get valid simplex weights
            raw = [kwargs[f'a{i}'] for i in range(num_clients)]
            total = sum(raw) if sum(raw) > 0 else 1.0
            alphas = [a / total for a in raw]

            # 2. Plain convex combination — NO Fisher
            theta_agg = torch.zeros(d, device=device)
            for i, alpha_i in enumerate(alphas):
                theta_agg += alpha_i * model_vectors[i]

            # 3. Evaluate on validation set
            candidate = copy.deepcopy(self.global_model)
            vector_to_parameters(theta_agg, candidate.parameters())
            acc, _ = evaluate_model(
                candidate, self.validation_dataset, device)
            return acc

        # Acquisition function
        if self.acq == 'ei':
            utility = acquisition.ExpectedImprovement(
                xi=self.xi, random_state=self.config['common'].get('seed', 0))
        elif self.acq == 'poi':
            utility = acquisition.ProbabilityOfImprovement(
                xi=self.xi, random_state=self.config['common'].get('seed', 0))
        else:  # default ucb
            utility = acquisition.UpperConfidenceBound(
                kappa=self.kappa,
                random_state=self.config['common'].get('seed', 0))

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            acquisition_function=utility,
            random_state=self.config['common'].get('seed', 0),
            verbose=2 if self.verbose else 0,
        )

        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

        # Reconstruct final model from best alpha*
        best_params = optimizer.max['params']
        raw_best = [best_params[f'a{i}'] for i in range(num_clients)]
        total_best = sum(raw_best) if sum(raw_best) > 0 else 1.0
        alpha_star = [a / total_best for a in raw_best]

        self.logger.info(f"FedFAW w/o Fisher optimal alpha*: {alpha_star}")

        theta_final = torch.zeros(d, device=device)
        for i, alpha_i in enumerate(alpha_star):
            theta_final += alpha_i * model_vectors[i]

        final_model = copy.deepcopy(self.global_model)
        vector_to_parameters(theta_final, final_model.parameters())
        return final_model
