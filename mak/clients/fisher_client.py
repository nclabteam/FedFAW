import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch import optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from mak.clients.base_client import BaseClient
from mak.utils.data import dataset_info, get_unique_classes, random_pertube
import copy
from tqdm import tqdm

from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag


class FisherClient(BaseClient):
    def __init__(self, client_id, strategy, train_data, logger, config, model, device, out_dir, num_classes,test_loader):
        super().__init__(client_id, strategy, train_data,
                         logger, config, model, device, out_dir, num_classes,test_loader)
        self.server_model = None
        self.local_epochs = self.config['client']['epochs']
        self.lr = self.config['client']['lr']
        if self.strategy == 'fisher_merge':
            self.fisher_type = 'diagonal'
        else:
            self.fisher_type = self.config['fisher']['fisher_type']



    def train(self):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(
        ), lr=self.config['client']['lr'], momentum=self.config['common']['sgd_momentum'])
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(self.config['client']['epochs']), desc=f'Local Training FedFisher client {self.client_id}'):
            epoch_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                if self.augmentation and self.transform_train is not None:
                    images = self.transform_train(images)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            if self.verbose:
                self.logger.info(
                    "Client %d completed epoch %d with average loss: %.4f",
                    self.client_id, epoch + 1, avg_loss
                )
        if self.config['client']['evaluate']:
            acc, loss = self.evaluate()
            self.logger.info(
                "Evaluate Client %d => Acc : %.4f Loss : %.4f",
                self.client_id, acc, loss
            )

        # client local training finished now calculate fisher information matrix
        if self.fisher_type == 'diagonal':
            fisher_matrix = self.calculate_diagonal_fisher_matrix()
        elif self.fisher_type == 'kfac':
            fisher_matrix = self.calculate_kfac_fisher_matrix()
        else:
            raise ValueError(f"Unknown fisher_type '{self.fisher_type}'. Expected 'diagonal' or 'kfac'.")

        return self.model.state_dict(), fisher_matrix


    def calculate_diagonal_fisher_matrix(self, model=None):
        # NNGeometry's Jacobian generator requires BatchNorm layers to be in eval mode.
        # If no external model is passed, use self.model and toggle eval/train in-place.
        model = model if model is not None else self.model
        model.eval()
        F_diagonal = FIM(model=model,
                         loader=self.train_loader,
                         representation=PMatDiag,
                         device=self.device,
                         n_output=self.num_classes)
        F_diagonal = F_diagonal.get_diag()
        model.train()
        return F_diagonal

    def calculate_kfac_fisher_matrix(self, model=None):
        # NNGeometry's Jacobian generator requires BatchNorm layers to be in eval mode.
        model = model if model is not None else self.model
        model.eval()
        F_kfac = FIM(model=model,
                     loader=self.train_loader,
                     representation=PMatKFAC,
                     device=self.device,
                     n_output=self.num_classes)
        model.train()
        return F_kfac
