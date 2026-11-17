import copy
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters, clip_grad_norm_
from mak.clients.base_client import BaseClient
from mak.utils.fedbens_utils import perturb_parameters
from tqdm import tqdm

try:
    from laplace import Laplace
except ImportError:
    Laplace = None

class FedBEnsClient(BaseClient):
    def __init__(self, client_id, strategy, train_data, logger, config, model, device, out_dir, num_classes, test_loader):
        super().__init__(client_id, strategy, train_data, logger, config, model, device, out_dir, num_classes, test_loader)
        
        self.n_mixtures = self.config.get('fedbens', {}).get('n_mixtures', 1)
        self.local_ens = self.config.get('fedbens', {}).get('local_ens', 1) # replicas per mixture component
        self.temperature = self.config.get('fedbens', {}).get('temperature', 0.1)
        self.prior_cov = self.config.get('fedbens', {}).get('prior_cov', 0.1)
        # Local training has no gradient clipping upstream (matching the
        # official implementation), so a client with a small/skewed data
        # shard (e.g. large num_clients) can diverge to NaN/Inf weights,
        # which then crashes laplace-torch's KFAC eigendecomposition
        # (CUSOLVER_STATUS_EXECUTION_FAILED / illegal memory access) with
        # no way to recover mid-process. grad_clip_norm prevents the
        # divergence at its source; max_train_retries is a backstop that
        # retries a diverged ensemble member from a fresh random init
        # before giving up and skipping it.
        self.grad_clip_norm = self.config.get('fedbens', {}).get('grad_clip_norm', 5.0)
        self.max_train_retries = self.config.get('fedbens', {}).get('max_train_retries', 3)


        if Laplace is None:
            self.logger.error("laplace-torch is not installed. FedBEns strategy will fail. Install it with 'pip install laplace-torch'.")

    def train(self, starting_points=None):
        if Laplace is None:
            raise ImportError("laplace-torch is required for FedBEns")

        # Mixture starting points, as in the official implementation
        # (main_FedBEns_kron.py:83-84 + train_model_kron.py:117-122): the
        # server draws n_mixtures random initializations ONCE and every
        # client starts mixture k from the SAME point k (random_init=False
        # on the clients). This shared start is what makes slot k comparable
        # across clients, which the server relies on (median per slot and
        # positional component alignment in negative_log_likelihood).
        # local_ens replicas of a mixture are (optionally) perturbed around
        # that shared starting point.
        #
        # We will train 'n_mixtures * local_ens' models in total.
        if starting_points is None:
            raise ValueError("FedBEnsClient.train() needs the server's shared starting_points")
        if len(starting_points) != self.n_mixtures:
            raise ValueError(f"Expected {self.n_mixtures} starting points, got {len(starting_points)}")

        all_params = []
        all_la = []

        total_models = self.n_mixtures * self.local_ens
        model_idx = 0

        for mixture_idx in range(self.n_mixtures):
            # Shared random starting point for this mixture component.
            base_net = copy.deepcopy(self.model).to(self.device)
            vector_to_parameters(starting_points[mixture_idx].to(self.device), base_net.parameters())

            for local_idx in range(self.local_ens):
                model_idx += 1
                self.logger.info(f"Client {self.client_id}: Training Ensemble Model {model_idx}/{total_models}")

                net = None
                for attempt in range(1, self.max_train_retries + 1):
                    # Create a clone for this ensemble member
                    candidate = copy.deepcopy(base_net).train()

                    # Perturb around the mixture's starting point if multiple
                    # replicas of it are requested
                    if self.local_ens > 1:
                        perturb_parameters(candidate, std=0.001)

                    if self._train_local_model(candidate):
                        net = candidate
                        break

                    # Retry from the same shared starting point (re-initializing
                    # locally would break cross-client slot alignment); data
                    # shuffling / augmentation make the retry differ.
                    self.logger.warning(
                        f"Client {self.client_id}: local training for ensemble member {model_idx} "
                        f"diverged to non-finite weights (attempt {attempt}/{self.max_train_retries}). "
                        f"Retrying from the same shared starting point."
                    )

                if net is None:
                    # Fail loudly instead of skipping: the server aligns mixture
                    # components across clients by position (median per slot,
                    # and fixed index arithmetic in negative_log_likelihood),
                    # so a client returning fewer members would either crash
                    # with an IndexError or silently mis-pair components.
                    msg = (f"Client {self.client_id}: ensemble member {model_idx} failed to reach a "
                           f"finite state after {self.max_train_retries} attempts. Consider lowering "
                           f"client.lr, lowering fedbens.grad_clip_norm, or raising max_train_retries.")
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                # Fit Laplace Approximation
                self.logger.info(f"Client {self.client_id}: Fitting Laplace for Model {model_idx}")
                la = Laplace(
                    net, 'classification',
                    subset_of_weights='all',
                    hessian_structure='kron',
                    prior_mean=0,
                    prior_precision=1.0 / self.prior_cov,
                    temperature=self.temperature
                )
                la.fit(self.train_loader)

                all_params.append(parameters_to_vector(net.parameters()).detach().cpu())
                all_la.append(la) # Note: Laplace objects can be large, might need careful handling or detachment

        return all_params, all_la

    def _train_local_model(self, net):
        """
        Trains `net` in place for the configured number of local epochs,
        clipping gradients to prevent divergence. Returns True if training
        finished with finite weights, False if the loss or weights went
        non-finite (NaN/Inf) at any point -- callers must not feed a
        False-returning model into Laplace.fit(), since a non-finite
        curvature matrix crashes laplace-torch's eigendecomposition
        (CUSOLVER_STATUS_EXECUTION_FAILED, then an unrecoverable illegal
        memory access on the CUDA context right after).
        """
        optimizer = torch.optim.SGD(net.parameters(),
                                   lr=self.config['client']['lr'],
                                   momentum=self.config['common'].get('sgd_momentum', 0.9))
        criterion = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.config['client']['epochs']):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                # Same on-tensor augmentation as the official FedBEns local
                # training (RandomCrop+Flip for CIFAR, see train_model_kron.py)
                # and as this repo's other strategies (BaseClient.train_fedavg).
                if self.augmentation and self.transform_train is not None:
                    images = self.transform_train(images)
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                if not torch.isfinite(loss):
                    return False
                loss.backward()
                clip_grad_norm_(net.parameters(), max_norm=self.grad_clip_norm)
                optimizer.step()

        return all(torch.isfinite(p).all() for p in net.parameters())

    def set_model(self, model):
        # Overriding set_model to ensure we keep the global model as a template
        self.model = model
