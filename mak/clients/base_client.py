import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


class BaseClient:
    def __init__(self, client_id, strategy, train_data, logger, config, model, device, out_dir, num_classes, test_loader = None):
        self.client_type = strategy.lower()
        self.client_id = client_id
        self.logger = logger
        self.strategy = strategy
        self.batch_size = config['client']['train_batch_size']
        self.train_data = train_data  # Already a Dataset or Subset
        self.data_size = len(train_data)
        self.train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.config = config
        self.out_dir = out_dir
        self.num_classes = num_classes

        self.verbose = config['common']['verbose']
        self.augmentation = config['client'].get('augmentation', False)
        
        # Initialize transformation for dynamic augmentation (on Tensors)
        dataset_name = config['common']['dataset'].lower()
        if dataset_name == 'cifar10' or dataset_name == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        elif dataset_name == 'fashion_mnist' or dataset_name == 'mnist':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
            ])
        else:
            self.transform_train = None

    def set_model(self, global_model):
        self.model.load_state_dict(global_model)
        if self.verbose:
            self.logger.info("Client %d set the global model.", self.client_id)

    def train(self):
        """Train the client's model on local data."""
        if self.client_type == 'fedprox':
            return self.train_fedprox()
        else:
            return self.train_fedavg()

    def train_fedavg(self):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(
        ), lr=self.config['client']['lr'], momentum=self.config['common']['sgd_momentum'])
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(self.config['client']['epochs']), desc=f'Local Training FedAvg client {self.client_id}'):
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
        return self.model.state_dict()

    def train_fedprox(self):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(
        ), lr=self.config['client']['lr'], momentum=self.config['common']['sgd_momentum'])
        criterion = torch.nn.CrossEntropyLoss()
        global_params = [val.detach().clone()
                         for val in self.model.parameters()]
        proximal_mu = self.config['fedprox']['proximal_mu']
        for epoch in tqdm(range(self.config['client']['epochs']), desc=f'Local Training FedProx client {self.client_id}'):
            epoch_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                if self.augmentation and self.transform_train is not None:
                    images = self.transform_train(images)
                optimizer.zero_grad()
                proximal_term = 0.0
                for local_weights, global_weights in zip(self.model.parameters(), global_params):
                    proximal_term += torch.square(
                        (local_weights - global_weights).norm(2))

                outputs = self.model(images)
                loss = criterion(outputs, labels) + \
                    (proximal_mu / 2) * proximal_term
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
        return self.model.state_dict()

    def evaluate(self):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy, avg_loss
