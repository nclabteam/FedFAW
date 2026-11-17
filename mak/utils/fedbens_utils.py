import torch
import torch.nn as nn
import copy
import torch.distributions as D
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from mak.utils.data import evaluate_model
from torch.utils.data import DataLoader

class kronecker_network_nb:
    def __init__(self, la, model_vector, device):
        # la is a Laplace object or similar with a .log_prob() method
        self.la = la
        self.model_param = model_vector
        self.device = device

    def log_pdf_network(self, x):
        """
        Compute the log PDF for the entire network.
        """
        # Ensure x is on the correct device and has the right shape if needed
        # Laplace's log_prob usually expects a parameter vector
        return self.la.log_prob(x)

class MixtureModel_kron_nb:
    def __init__(self, means, la_list, model_template, device, weights, mean_prior, cov_prior, n_clients, n_client_mixtures, logger, val_at_server=False):
        """
        Initialize mixture on kronecker networks.
        
        means: list of flattened parameter vectors (state_dicts converted)
        la_list: list of Laplace objects
        """
        self.means = [torch.as_tensor(mu, dtype=torch.float32).to(device) for mu in means]
        self.la = la_list
        self.num_components = len(means)
        self.device = device
        self.prior_mean = mean_prior.to(device)
        self.prior_cov = cov_prior.to(device)
        self.n_clients = n_clients
        self.num_client_mix = n_client_mixtures
        self.model_template = model_template
        self.val_at_server = val_at_server
        self.logger = logger

        # Prior distribution
        self.prior = D.Independent(D.Normal(self.prior_mean, torch.sqrt(self.prior_cov)), 1)

        self.component_distributions = []
        for i in range(self.num_components):
            local = kronecker_network_nb(self.la[i], self.means[i], device)
            self.component_distributions.append(local)

    def negative_log_likelihood(self, point):
        log_probs_clients = []
        for j in range(int(self.n_clients)):
            aux = []
            for i in range(int(self.num_client_mix)):
                index = int((j * self.num_client_mix) + i)
                # Evaluates log_prob of the mixture component
                log_prob = self.component_distributions[index].log_pdf_network(point)
                aux.append(log_prob)
            
            # Mixture over local ensemble members per client
            log_probs_clients.append(torch.logsumexp(torch.stack(aux), dim=0))

        # Sum of log-likelihoods over clients
        negative_log_likelihood = - torch.sum(torch.stack(log_probs_clients), dim=0)
        return negative_log_likelihood

    def negative_log_posterior(self, point):
        nll = self.negative_log_likelihood(point)
        # Bayesian ensemble combination logic: log P(θ|D) ∝ Σ log P(θ|D_i) - (N-1) log P(θ)
        log_prior = self.prior.log_prob(point)
        total = nll + (self.n_clients - 1) * log_prior
        return total

    def optimize(self, l_r, n_steps, w_0, net, dataset_val, config, logger):
        x = w_0.clone().detach().to(self.device)
        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=l_r)
        
        test_acc_max = 0.0
        train_loss_min = float('inf')
        best_parameters_train = x.detach().clone()
        best_parameters_val = x.detach().clone()

        net_glob_copy = copy.deepcopy(net).to(self.device)
        
        val_loader = None
        if dataset_val is not None:
            val_loader = DataLoader(dataset_val, batch_size=config['client']['test_batch_size'], shuffle=False)

        for k in range(n_steps):
            optimizer.zero_grad()
            loss = self.negative_log_posterior(x)
            loss.backward()
            
            # Stabilization
            torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
            optimizer.step()

            if loss < train_loss_min:
                train_loss_min = loss.item()
                best_parameters_train = x.detach().clone()

            if (k % 30 == 0 or k == n_steps - 1):
                vector_to_parameters(x, net_glob_copy.parameters())
                if val_loader:
                    acc, _ = evaluate_model(net_glob_copy, val_loader, self.device)
                    if acc > test_acc_max:
                        test_acc_max = acc
                        best_parameters_val = x.detach().clone()
                    logger.info(f"Step {k}: Loss: {loss.item():.4f}, Val Acc: {acc:.4f}")
                else:
                    logger.info(f"Step {k}: Loss: {loss.item():.4f}")

        if self.val_at_server and val_loader:
            self.logger.info("Best parameters selected based on validation accuracy")
            return best_parameters_val
        else:
            self.logger.info("Best parameters selected based on training loss")
            return best_parameters_train

def instantiate_model(flattened_params, model_instance):
    param_pos = 0
    for name, param in model_instance.named_parameters():
        param_size = param.numel()
        param_data = flattened_params[param_pos:param_pos + param_size].view(param.size())
        param.data.copy_(param_data)
        param_pos += param_size
    return model_instance

def reset_parameters(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()

def perturb_parameters(model, std=0.001):
    device = next(model.parameters()).device
    for param in model.parameters():
        param.data += torch.randn(param.size(), device=device) * std
