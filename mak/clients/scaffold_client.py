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
from mak.optimizers import SCAFFOLDOptimizer


class ScaffoldClient(BaseClient):
    def __init__(self, client_id, strategy, train_data, logger, config, model, device, out_dir, num_classes,test_loader):
        super().__init__(client_id, strategy, train_data,
                         logger, config, model, device, out_dir, num_classes,test_loader)
        self.y = None
        # delta_y & delta_c of a client are communicated to the central server after client_update has completed
        self.delta_y = None
        self.delta_c = None
        self.server_cv = None
        self.client_cv = []
        for param in self.model.parameters():
            self.client_cv.append(torch.zeros(param.shape))

        self.num_batches = len(self.train_loader)
        self.server_model = None
        self.local_epochs = self.config['client']['epochs']
        self.lr = self.config['client']['lr']

    def train(self):
        self.server_model = copy.deepcopy(self.model)
        self.model.train()
        optimizer = SCAFFOLDOptimizer(
            params=self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(self.local_epochs), desc=f'Local Training Scaffold client {self.client_id}'):
            epoch_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # update using scaffold optimizer
                optimizer.step(self.server_cv, self.client_cv)
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            if self.verbose:
                self.logger.info(
                    "Client %d completed epoch %d with average loss: %.4f",
                    self.client_id, epoch + 1, avg_loss
                )

        # scaffold extra starts
        self.server_cv = [cv.to(self.device) for cv in self.server_cv]
        self.client_cv = [cv.to(self.device) for cv in self.client_cv]

        for ci, c, x, yi in zip(self.client_cv, self.server_cv, self.server_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1/self.num_batches / \
                self.local_epochs/self.lr * (x - yi)
        if self.config['client']['evaluate']:
            acc, loss = self.evaluate()
            self.logger.info(
                "Evaluate Client %d => Acc : %.4f Loss : %.4f",
                self.client_id, acc, loss
            )

        return self.model.state_dict()

    def get_delta_yc(self):
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.server_cv, self.server_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1/self.num_batches /
                           self.local_epochs/self.lr * (x - yi))

        return delta_y, delta_c
