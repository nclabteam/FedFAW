import copy
import torch
from mak.servers.base_server import FedAvgServer
from mak.utils.fedbens_utils import MixtureModel_kron_nb, instantiate_model, reset_parameters
from mak.utils.comm_utils import get_size_mb
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class FedBEnsServer(FedAvgServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset):
        super().__init__(model, config, logger, device, out_dir, validation_dataset)
        self.logger.info("Server initialized: FedBEns Server")
        
        self.n_mixtures = self.config.get('fedbens', {}).get('n_mixtures', 1)
        self.local_ens = self.config.get('fedbens', {}).get('local_ens', 1)
        self.total_mix_per_client = self.n_mixtures * self.local_ens
        
        # Prior parameters
        d = parameters_to_vector(self.global_model.parameters()).numel()
        self.prior_mean = torch.zeros(d).to(self.device)
        self.prior_cov = torch.full((d,), self.config.get('fedbens', {}).get('prior_cov', 0.1)).to(self.device)
        self.ensemble_models = []
        self.starting_points = None

    def _get_starting_points(self):
        """
        n_mixtures random initializations, drawn once on the server and shared
        by all clients (official main_FedBEns_kron.py:83-84:
        `[copy.deepcopy(net_glob).apply(reset) for _ in range(N_ENSEMBLE)]`).
        Uses the torch RNG, so it is controlled by the run seed.
        """
        if self.starting_points is None:
            self.starting_points = []
            for _ in range(self.n_mixtures):
                net = copy.deepcopy(self.global_model)
                net.apply(reset_parameters)
                self.starting_points.append(parameters_to_vector(net.parameters()).detach().clone())
        return self.starting_points

    def fit(self, clients, round):

        results = {}
        upload_cost = 0
        download_cost = 0
        
        global_model_size = sum(p.numel() * 4 for p in self.global_model.parameters()) / (1024 * 1024)
        starting_points = self._get_starting_points()

        for c_id in clients:
            # Download cost: every client receives the n_mixtures shared
            # starting points (each the size of the model). The global model
            # itself is only used as an architecture template, so it is not
            # counted separately. (A shared-seed scheme could cut this to ~0,
            # but we count the vectors as transmitted.)
            download_cost += global_model_size * len(starting_points)
            client = clients[c_id]
            client.set_model(copy.deepcopy(self.global_model))

            # Client returns list of weights and list of Laplace objects
            weights_list, la_list = client.train(starting_points=starting_points)

            # Upload cost: model weights AND the fitted KFAC Laplace
            # posteriors. The posteriors are genuinely computed client-side
            # (not a server-side simulation shortcut) and are essential
            # inputs to the server's aggregation (MixtureModel_kron_nb calls
            # la.log_prob() on each one) -- a real deployment needs clients
            # to transmit them too. Measured ~4x the size of the model
            # itself for this project's FedNet architecture; previously not
            # counted at all, undercounting FedBEns's real upload cost ~5x.
            upload_cost += (sum(w.numel() * 4 for w in weights_list) / (1024 * 1024))
            upload_cost += sum(get_size_mb(la) for la in la_list)
            
            results[c_id] = {
                'weights': weights_list,
                'laplace': la_list,
                'num_samples': client.data_size
            }
            
        return results, upload_cost, download_cost

    def aggregate(self, results):
        self.logger.info("FedBEns: Constructing global mixture model...")
        
        all_means = []
        all_la = []
        starting_points = []
        
        client_ids = sorted(results.keys())
        n_clients = len(client_ids)
        
        for c_id in client_ids:
            all_means.extend(results[c_id]['weights'])
            all_la.extend(results[c_id]['laplace'])
        
        # Calculate median starting points as in user code
        # We take the median of weights across clients for each ensemble slot
        # Assuming all clients returned the same number of ensemble members
        num_mix = self.total_mix_per_client
        
        # Each mixture j from clients i
        medians = []
        for j in range(num_mix):
            slot_weights = torch.stack([results[c_id]['weights'][j] for c_id in client_ids])
            medians.append(torch.median(slot_weights, dim=0).values)
        
        # Weights for mixtures (uniform in the snippet)
        weights = torch.ones(len(all_means)).to(self.device)
        
        mixture_model = MixtureModel_kron_nb(
            means=all_means,
            la_list=all_la,
            model_template=self.global_model,
            device=self.device,
            weights=weights,
            mean_prior=self.prior_mean,
            cov_prior=self.prior_cov,
            n_clients=n_clients,
            n_client_mixtures=num_mix,
            val_at_server=self.config.get('fedbens', {}).get('val_at_server', True),
            logger=self.logger
        )
        
        # Optimize all mixture components to build the final ensemble
        # In Bayesian Ensemble, we want multiple diverse models representing the posterior
        self.logger.info(f"FedBEns: Optimizing {len(medians)} ensemble members...")
        
        lr = self.config.get('fedbens', {}).get('lr_opt', 0.001)
        steps = self.config.get('fedbens', {}).get('steps_opt', 300)
        
        self.ensemble_models = []
        self.w_hats = []
        for i, start_p in enumerate(medians):
            self.logger.info(f"FedBEns: Optimizing member {i+1}/{len(medians)}...")
            best_w = mixture_model.optimize(
                l_r=lr,
                n_steps=steps,
                w_0=start_p,
                net=self.global_model,
                # self.validation_dataset is the raw dataset_val from
                # apply_val_data_shift (FedBEnsServer never wraps it in a
                # DataLoader, unlike e.g. CMAESMergeServer/FedFAWServer,
                # where .dataset correctly retrieves the DataLoader's
                # wrapped dataset). Passing it directly here is correct --
                # optimize() wraps it in its own DataLoader internally.
                # ".dataset" here was accidentally reaching past it to the
                # *unfiltered* underlying full dataset for Subset-based
                # shifts (label/domain/none), and crashing outright for
                # CovariateShiftSubset (which has no .dataset attribute).
                dataset_val=self.validation_dataset if self.validation_dataset else None,
                config=self.config,
                logger=self.logger
            )
            # Create a dedicated model instance for this ensemble member
            model_member = copy.deepcopy(self.global_model)
            self.w_hats.append(best_w)
            instantiate_model(best_w, model_member)
            self.ensemble_models.append(model_member)

        #added this as i feel this was missing from the original implementation
        self.ensemble_models = [instantiate_model(W, copy.deepcopy(self.global_model).apply(reset_parameters)) for W in self.w_hats]
        
        # For compatibility with standard FL loops that use global_model,
        # we set global_model to the first optimized member.
        # But for FedBEns, the "real" result is the ensemble.
        if self.ensemble_models:
            self.global_model.load_state_dict(self.ensemble_models[0].state_dict())
            
        self.logger.info("FedBEns aggregation completed.")


    def evaluate(self, test_loader, round_num):
        """Override evaluate to perform ensemble evaluation if ensemble exists."""
        if not self.ensemble_models:
            return super().evaluate(test_loader, round_num)
        
        from mak.utils.data import evaluate_ensemble
        self.logger.info(f"FedBEns: Evaluating ensemble of {len(self.ensemble_models)} models...")
        accuracy, average_loss = evaluate_ensemble(self.ensemble_models, test_loader, self.device, self.num_classes)
        
        self.logger.info(
            "Ensemble Accuracy: %.2f%%, Loss: %.4f", accuracy, average_loss)
            
        return accuracy, average_loss

    def __repr__(self) -> str:
        return " FedBEns Strategy"

