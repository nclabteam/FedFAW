import os
import json
import time
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from mak.servers.fedfisher_server import FedFisherServer
from mak.utils.fedfaw_utils import FedFAW
from mak.utils.data import dataset_info
from mak.utils.dense_utils import Generator, AdvSynthesizer, Ensemble, LabeledSyntheticDataset
from torch.utils.data import DataLoader

class FedFAWServer(FedFisherServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset, full_dataset, client_indices):
        super().__init__(model, config, logger, device, out_dir, validation_dataset, full_dataset, client_indices)
        self.logger.info(f"Server initialized: FedFAW Server : Fisher Type : {self.fisher_type}")
        
        self._faw_summary = None
        faw_cfg = self.config.get('fedfaw', {})
        self.use_synthetic_val = faw_cfg.get('use_synthetic_val', False)
        
        if self.use_synthetic_val:
            self.nz = self.config.get('dense', {}).get('nz', 256)
            img_size = self.im_size[0]
            self.im_size_dims = (self.channel, img_size, img_size)
            self.generator = Generator(nz=self.nz, ngf=64, img_size=img_size, nc=self.channel).to(self.device)
            self.synthetic_batches = faw_cfg.get('synthetic_val_batches', 4) # Number of batches to generate
            self.logger.info(f"FedFAW: Data-Free mode enabled. Generator initialized. Will generate {self.synthetic_batches} batches.")

    def __repr__(self) -> str:
        return f" FedFAW Strategy : Fisher Type : {self.fisher_type}"

    def aggregate(self, results):
        # Server time = everything the server does after receiving client uploads
        # (BO search + final merge). Client training / Fisher computation excluded.
        t_server = time.perf_counter()
        self._aggregate(results)
        server_time = time.perf_counter() - t_server
        self.logger.info(f"FedFAW server time (s): {server_time:.2f}")
        if self._faw_summary is not None:
            self._faw_summary['server_time_s'] = server_time
            with open(os.path.join(self.out_dir, 'bo_summary.json'), 'w') as f:
                json.dump(self._faw_summary, f)

    def _aggregate(self, results):
        # We need both diagonal and kfac lists for FedFAW
        total_samples = sum(entry['num_samples'] for entry in results.values())
        model_vectors = []
        F_diag_list = []
        F_kfac_list = []
        models = []

        weights_list = []
        
        # Sort results by key to ensure consistency
        sorted_keys = sorted(results.keys())
        
        for c_id in sorted_keys:
            client_data = results[c_id]
            client_model = client_data['model']
            data_size = client_data['num_samples']
            fisher_result = client_data['fisher_matrix']

            # Create model object
            client_model_obj = copy.deepcopy(self.global_model)
            client_model_obj.load_state_dict(client_model)
            models.append(client_model_obj)
            model_vectors.append(parameters_to_vector(client_model_obj.parameters()))

            weight = data_size / total_samples
            weights_list.append(weight)

            # Store Fisher matrices correctly
            if self.fisher_type == 'diagonal':
                F_diag_list.append(fisher_result)
                F_kfac_list.append(None)
            else: # kfac
                F_kfac_list.append(fisher_result)
                F_diag_list.append(None)

        # 2. Check for synthetic validation requirement
        dataset_val = self.validation_dataset.dataset
        if self.use_synthetic_val:
            self.logger.info("Generating synthetic validation dataset for FedFAW...")
            ensemble_model = Ensemble(models, weights_list).to(self.device)
            save_dir = os.path.join(self.out_dir, "faw_synthetic_pool")
            
            # Setup DENSE-style generator
            # We use a higher iteration count for FedFAW validation data to ensure quality
            faw_cfg = self.config.get('fedfaw', {})
            synth_iters = faw_cfg.get('synthetic_val_iters', 100)
            
            synthesizer = AdvSynthesizer(
                teacher=ensemble_model,
                student=self.global_model,
                generator=self.generator,
                nz=self.nz,
                num_classes=self.num_classes,
                img_size=self.im_size_dims,
                iterations=synth_iters,
                lr_g=self.config.get('dense', {}).get('lr_g', 1e-3),
                synthesis_batch_size=self.config.get('dense', {}).get('synthesis_batch_size', 256),
                sample_batch_size=self.config.get('dense', {}).get('sample_batch_size', 256),
                adv=0.5, bn=1.0, oh=1.0,
                save_dir=save_dir,
                dataset=self.config['common']['dataset'].lower(),
                mean=self.mean,
                std=self.std
            )
            
            # Synthesize multiple batches to ensure diversity and class coverage
            for b in range(self.synthetic_batches):
                self.logger.info(f"Synthesizing batch {b+1}/{self.synthetic_batches}...")
                synthesizer.synthesize(b)
            
            # Create LabeledSyntheticDataset
            dataset_val = LabeledSyntheticDataset(
                root=save_dir,
                transform=synthesizer.transform,
                nc=self.channel,
                teacher=ensemble_model,
                device=self.device
            )
            self.logger.info(f"Synthetic validation dataset created with {len(dataset_val)} samples.")

        # Call the FedFAW logic
        updated_model = self.get_fedfaw_model(
            self.d, len(models), weights_list, self.config, self.global_model,
            models, model_vectors, F_kfac_list, F_diag_list,
            dataset_val
        )

        # Update the global model parameters
        self.global_model.load_state_dict(updated_model.state_dict())
        self.logger.info("FedFAW model aggregation completed.")

    def get_fedfaw_model(
        self, d, n, p, args, net_glob, models, model_vectors, F_kfac_list, F_diag_list, 
        dataset_val, seed=0
    ):
        """
        Implementation of the Federated Bayesian Adaptive Fisher Aggregation (FedFAW).
        """
        use_kfac = (self.fisher_type == 'kfac')
        
        # Default parameters
        init_points = 10
        n_iter = 50
        acq = 'ucb'
        kappa = 2.576
        xi = 0.01

        # Pull parameters from the 'fedfaw' block in config
        if 'fedfaw' in self.config:
            faw_cfg = self.config['fedfaw']
            init_points = faw_cfg.get('init_points', init_points)
            n_iter = faw_cfg.get('n_iter', n_iter)
            acq = faw_cfg.get('acquisition_fn', acq)
            kappa = faw_cfg.get('kappa', kappa)
            xi = faw_cfg.get('xi', xi)

        seed = self.config['common'].get('seed', seed)

        faw = FedFAW(
            base_model=net_glob,
            models=models,
            model_vectors=model_vectors,
            F_kfac_list=F_kfac_list,
            F_diag_list=F_diag_list,
            use_kfac=use_kfac,
        )
        
        merged_model = faw.merge(
            dataset_val=dataset_val,
            args=args,
            acquisition_fn=acq,
            seed=seed,
            init_points=init_points,
            n_iter=n_iter,
            kappa=kappa,
            xi=xi,
            verbose=self.verbose,
        )
        
        optimal_coef = faw.get_optimal_coefficients()
        self.logger.info(f"Optimal coefficients: {optimal_coef}")
        self.logger.info(f"FedFAW val acc of alpha*: {faw.best_val_acc:.4f} "
                         f"({len(faw.history)} evaluations, BO time {faw.bo_time:.2f}s)")

        # Logging only: fingerprint of the client models/Fishers, so runs that should
        # share identical client models (same K, seed; different BO budget) can be checked.
        fp = lambda ts: [float(t.double().sum()) for t in ts]
        s = sum(optimal_coef) if sum(optimal_coef) > 0 else 1.0
        self._faw_summary = {
            'num_clients': n, 'init_points': init_points, 'n_iter': n_iter,
            'acq': acq, 'kappa': kappa, 'seed': seed,
            'val_acc_alpha_star': faw.best_val_acc,
            'alpha_star_raw': optimal_coef,
            'alpha_star_normalized': [a / s for a in optimal_coef],
            'bo_time_s': faw.bo_time,
            'history': faw.history,
            'client_model_fingerprint': fp(model_vectors),
            'client_fisher_fingerprint': fp(F_diag_list) if not use_kfac else None,
        }

        return merged_model
