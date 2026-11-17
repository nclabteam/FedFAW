import copy
import torch
import torch.nn as nn
from mak.clients.base_client import BaseClient
from mak.utils.fedlpa_utils import KFACOptimizer, posterior_loss
from collections import OrderedDict
from tqdm import tqdm

class FedLPAClient(BaseClient):
    def __init__(self, client_id, strategy, train_data, logger, config, model, device, out_dir, num_classes, test_loader):
        super().__init__(client_id, strategy, train_data, logger, config, model, device, out_dir, num_classes, test_loader)
        
        self.mu_global = None
        self.f_global = None
        self.f_local = OrderedDict()
        
        # Initialize f_local structure
        for name, module in self.model.named_modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                self.f_local[name] = [None, None]

    def set_global_info(self, mu_global, f_global):
        self.mu_global = mu_global
        self.f_global = f_global

    def train(self):
        self.model.to(self.device).train()
        
        # Use SGD as in the snippet
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['client']['lr'], 
                                   momentum=self.config['common'].get('sgd_momentum', 0.9), 
                                   weight_decay=self.config['client'].get('weight_decay', 1e-5))
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        eta = self.config.get('fedlpa', {}).get('eta', 1.0)
        
        for epoch in range(self.config['client']['epochs']):
            epoch_loss = 0
            epoch_p_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Posterior loss term (Alignment)
                # Only apply if we have global info and it's not the first round
                if self.mu_global is not None and self.f_global is not None:
                    # Check if global FIM is populated
                    sample_layer = next(iter(self.f_local))
                    if self.f_global[sample_layer][0] is not None:
                        p_loss = eta * posterior_loss(self.f_global, self.f_local, self.mu_global, self.model, self.device)
                        loss += p_loss
                        epoch_p_loss += p_loss.item()
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # self.logger.info(f"Client {self.client_id} Epoch {epoch+1} Loss: {epoch_loss/len(self.train_loader):.4f}")

        # Post-training pass: Collect KFAC statistics (Empirical Fisher)
        self.logger.info(f"Client {self.client_id}: Collecting KFAC statistics...")
        self.calculate_kfac_stats()
        
        return self.model.state_dict(), self.f_local

    def calculate_kfac_stats(self):
        self.model.eval()
        kfac_optimizer = KFACOptimizer(self.model)
        kfac_optimizer.acc_stats = True
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        # Reset local Fisher for new round collection
        # We'll re-populate but we need to keep the structure
        new_f_local = OrderedDict()
        for name in self.f_local:
            new_f_local[name] = [None, None]

        data_size = len(self.train_loader.dataset)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            kfac_optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            for module in kfac_optimizer.modules:
                name = kfac_optimizer.module2name[module]
                batch_weight = len(labels) / data_size
                
                A = kfac_optimizer.m_aa[module].detach()
                G = kfac_optimizer.m_gg[module].detach()
                
                if new_f_local[name][0] is None:
                    new_f_local[name][0] = batch_weight * A
                    new_f_local[name][1] = batch_weight * G
                else:
                    new_f_local[name][0] += batch_weight * A
                    new_f_local[name][1] += batch_weight * G
            
            # Limit pass if dataset is too large
            if (batch_idx + 1) * len(labels) >= 1024:
                 break

        
        # Remove hooks
        for hook in kfac_optimizer.hooks:
            hook.remove()
        
        # Add damping (I) to local Fisher as per snippet
        # This helps with numerical stability and is part of the algorithm
        coordinator = self.config.get('fedlpa', {}).get('coordinator', 0.0001)
        coor_term = torch.sqrt(torch.tensor(1.0 - coordinator))
        
        for name in new_f_local:
            if new_f_local[name][0] is not None:
                pi = 1.0 # default balance
                I_A = torch.eye(new_f_local[name][0].shape[0]).to(self.device)
                I_G = torch.eye(new_f_local[name][1].shape[0]).to(self.device)
                new_f_local[name][0] += pi * coor_term * I_A
                new_f_local[name][1] += (1.0 / pi) * coor_term * I_G
        
        self.f_local = new_f_local
