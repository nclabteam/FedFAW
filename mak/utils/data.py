import numpy as np
import torch
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from mak.datasets.gender_classification import CelebAGenderDataset


dataset_info = {
    'cifar10': {
        'num_classes': 10,
        'channel': 3,
        'img_size': (32, 32),
        'mean': [0.491, 0.482, 0.447],
        'std': [0.247, 0.243, 0.262],
    },
    'cifar100': {
        'num_classes': 100,
        'channel': 3,
        'img_size': (32, 32),
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
    },
    'mnist': {
        'num_classes': 10,
        'channel': 1,
        'img_size': (28, 28),
        'mean': [0.1307],
        'std': [0.3081],
    },
    'fashion_mnist': {
        'num_classes': 10,
        'channel': 1,
        'img_size': (28, 28),
        'mean': [0.1307],
        'std': [0.3081],
    },
    'celeba': {
        'num_classes': 2,
        'channel': 3,
        'img_size': (64, 64),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    },
    'stl': {
        'num_classes': 10,
        'channel': 3,
        'img_size': (32, 32),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    },
    'svhn': {
        'num_classes': 10,
        'channel': 3,
        'img_size': (32, 32),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    },
    'cinic10': {
        'num_classes': 10,
        'channel': 3,
        'img_size': (32, 32),
        'mean': [0.47889522, 0.47227842, 0.43047404],
        'std': [0.24205776, 0.23828046, 0.25874835],
    },
    'gtsrb': {
        'num_classes': 43,
        'channel': 3,
        'img_size': (32, 32),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    },
    'domainnet': {
            'num_classes': 10,
            'channel': 3,
            'img_size': (64, 64),
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
    'uci_har': {
        # Tabular sensor-feature dataset (not an image): no channel/img_size/
        # mean/std, since those only drive torchvision transforms for the
        # image datasets above. input_dim is the flat feature-vector length
        # (561), consumed by the MLP model instead. Its path is hardcoded in
        # mak/datasets/uci_har.py (DATASET_PATH), matching where
        # scripts/download_process_har.py saves it.
        'num_classes': 6,
        'input_dim': 561,
    },
    'capture24': {
        # Tabular, like uci_har: one feature vector per 10 s accelerometer
        # window (46 features, see mak/datasets/capture24.py), produced by
        # scripts/download_process_capture24.py. num_classes is for the
        # Willetts2018 label scheme (sleep, sit-stand, vehicle, mixed-activity,
        # walking, bicycling).
        'num_classes': 6,
        'input_dim': 46,
    },
}


def load_data(logger, dataset_name, dir_alpha, total_clients, test_batch_size):
    """Load and partition data among clients using Dirichlet distribution for non-IID setting."""

    if dataset_name not in dataset_info:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_cfg = dataset_info[dataset_name]
    logger.info(
        f"Loading Dataset : {dataset_name} using Dir(alpha) = {dir_alpha}")

    # Define transforms
    if dataset_name in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_cfg['mean'], dataset_cfg['std'])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_cfg['mean'], dataset_cfg['std'])
        ])
    elif dataset_name == 'celeba':
        train_transform = transforms.Compose([
            transforms.Resize(dataset_cfg['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_cfg['mean'], dataset_cfg['std'])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(dataset_cfg['img_size']),
            transforms.CenterCrop(dataset_cfg['img_size'][0]),
            transforms.ToTensor(),
            transforms.Normalize(dataset_cfg['mean'], dataset_cfg['std'])
        ])
    else:
        # For MNIST, FashionMNIST: No augmentation by default
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_cfg['mean'], dataset_cfg['std'])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_cfg['mean'], dataset_cfg['std'])
        ])

    # Load datasets with respective transforms
    if dataset_name == 'cifar10':
        full_train_dataset = datasets.CIFAR10(
            root='./datasets', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(
            root='./datasets', train=False, download=True, transform=test_transform)
    elif dataset_name == 'cifar100':
        full_train_dataset = datasets.CIFAR100(
            root='./datasets', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(
            root='./datasets', train=False, download=True, transform=test_transform)
    elif dataset_name == 'mnist':
        full_train_dataset = datasets.MNIST(
            root='./datasets', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(
            root='./datasets', train=False, download=True, transform=test_transform)
    elif dataset_name == 'fashion_mnist':
        full_train_dataset = datasets.FashionMNIST(
            root='./datasets', train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(
            root='./datasets', train=False, download=True, transform=test_transform)
    elif dataset_name == 'celeba':
        full_train_dataset = CelebAGenderDataset(
            root="./datasets", split="train", transform=train_transform, download=True)
        test_dataset = CelebAGenderDataset(
            root="./datasets", split="test", transform=test_transform, download=True)
    elif dataset_name == "stl":
        full_train_dataset = datasets.STL10(
            root="./datasets", split="train", download=True, transform=train_transform)
        test_dataset = datasets.STL10(
            root="./datasets", split="test", download=True, transform=test_transform)

    # Split data using Dirichlet distribution
    targets = np.array(full_train_dataset.targets if hasattr(
        full_train_dataset, 'targets') else full_train_dataset.labels)
    client_indices = partition_data_min(
        logger=logger, targets=targets, total_clients=total_clients, dir_alpha=dir_alpha)
    logger.info("Data partitioned among %d clients.", total_clients)

    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False)

    return full_train_dataset, client_indices, test_loader


def partition_data_min(logger, targets, total_clients, dir_alpha, min_samples=10):
    """Partition data using Dirichlet distribution, ensuring each client has at least min_samples."""
    num_classes = np.max(targets) + 1
    n_samples = len(targets)

    # Check feasibility
    if total_clients * min_samples > n_samples:
        raise ValueError(
            f"Cannot assign {min_samples} samples to each of {total_clients} clients with only {n_samples} total samples")

    client_indices = [[] for _ in range(total_clients)]

    # Step 1: Assign minimum samples to each client
    all_indices = np.arange(n_samples)
    np.random.shuffle(all_indices)
    min_assignment = np.array_split(
        all_indices[:total_clients * min_samples], total_clients)
    for client_idx, indices in enumerate(min_assignment):
        client_indices[client_idx].extend(indices)

    # Remaining indices to distribute
    remaining_indices = all_indices[total_clients * min_samples:]
    remaining_targets = targets[remaining_indices]

    # Step 2: Distribute remaining samples using Dirichlet
    if len(remaining_indices) > 0:
        for c in range(num_classes):
            idx_c = np.where(remaining_targets == c)[0]
            if len(idx_c) > 0:
                np.random.shuffle(idx_c)
                proportions = np.random.dirichlet(
                    np.repeat(dir_alpha, total_clients))
                split_points = (np.cumsum(proportions) *
                                len(idx_c)).astype(int)[:-1]
                client_splits = np.split(idx_c, split_points)
                for i, split in enumerate(client_splits):
                    client_indices[i].extend(remaining_indices[split])

    # Shuffle each client's indices
    for i in range(total_clients):
        np.random.shuffle(client_indices[i])

    # Verify minimum samples
    for i, indices in enumerate(client_indices):
        if len(indices) < min_samples:
            raise RuntimeError(
                f"Client {i} has {len(indices)} samples, less than required {min_samples}")

    return client_indices


def partition_data(logger, targets, total_clients, dir_alpha):
    """Partition data using Dirichlet distribution."""
    num_classes = np.max(targets) + 1
    client_indices = [[] for _ in range(total_clients)]
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(dir_alpha, total_clients))
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        client_splits = np.split(idx_c, proportions)
        for i, split in enumerate(client_splits):
            client_indices[i].extend(split)
    for i in range(total_clients):
        np.random.shuffle(client_indices[i])
    return client_indices


def evaluate_model(model, test_loader, device):
    """
    Evaluates the model's performance on the test dataset.

    Args:
        model (torch.nn.Module): The global model to evaluate.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to evaluate on.
    """
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():  # No gradients needed
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = test_loss / total

    return accuracy, average_loss


def evaluate_ensemble(models, test_loader, device, num_classes):
    """
    Evaluates the ensemble performance on the test dataset by averaging probabilities.
    Args:
        models (list of torch.nn.Module): The ensemble models to evaluate.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to evaluate on.
        num_classes (int): Number of classes for the dataset being evaluated.
    """
    for m in models:
        m.eval()
        m.to(device)

    total = 0
    test_loss = 0
    correct = 0
    disagreement = 0
    predss=[]
    groud_truth=[]

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            ensemble_probs = torch.zeros((inputs.size(0), num_classes)).to(device)

            model_preds = []
            for model in models:
                log_probs = model(inputs)
                probs = F.softmax(log_probs, dim=1)
                ensemble_probs += probs

                _,pred_class = probs.max(1)
                model_preds.append(pred_class)
            
            ensemble_probs = ensemble_probs / len(models)
            
            test_loss += F.cross_entropy(torch.log(ensemble_probs), labels, reduction='sum').item()
            
            # Calculate the number of correct predictions
            _, y_pred = ensemble_probs.max(1)
            total += labels.size(0)
            correct += y_pred.eq(labels).sum().item()

            predss.append(y_pred)
            groud_truth.append(labels)

            # Disagreement
            model_preds = torch.stack(model_preds,dim=1)
            # disagreement += torch.any(model_preds != model_preds[:,0].unsqueeze(1),dim=1).sum().item()

    # Calculate average test loss and accuracy
    test_loss /= total
    accuracy = 100.0 * correct / total
    # disagreement_ratio= 100.*disagreement/len(datatest)
    target= torch.cat(groud_truth)
    y_pred= torch.cat(predss)

    return accuracy, test_loss


# def evaluate_ensemble(models, test_loader, device):
#     """
#     Evaluates the ensemble performance on the test dataset by averaging probabilities.
#     Args:
#         models (list of torch.nn.Module): The ensemble models to evaluate.
#         test_loader (DataLoader): DataLoader for test data.
#         device (torch.device): Device to evaluate on.
#     """
#     for m in models:
#         m.eval()
#         m.to(device)
    
#     total = 0
#     correct = 0
#     test_loss = 0.0
    
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             ensemble_probs = 0
#             for model in models:
#                 outputs = model(inputs)
#                 ensemble_probs += torch.softmax(outputs, dim=1)
            
#             ensemble_probs = ensemble_probs / len(models)
            
#             # Use negative log likelihood of the average probability for loss
#             loss = F.nll_loss(torch.log(ensemble_probs + 1e-12), labels)
            
#             test_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(ensemble_probs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     average_loss = test_loss / total

#     return accuracy, average_loss



def get_unique_classes(dataloader):
    all_labels = []
    for _, labels in dataloader:
        # Assuming labels are in tensor format
        all_labels.extend(labels.numpy())

    unique_classes = list(set(all_labels))
    return unique_classes


def random_pertube(model, gamma):
    new_model = copy.deepcopy(model)
    for p in new_model.parameters():
        gauss = torch.normal(mean=torch.zeros_like(p), std=1)
        if p.grad is None:
            p.grad = gauss
        else:
            p.grad.data.copy_(gauss.data)

    norm = torch.norm(torch.stack(
        [p.grad.norm(p=2) for p in new_model.parameters() if p.grad is not None]), p=2)

    with torch.no_grad():
        scale = gamma / (norm + 1e-12)
        scale = torch.clamp(scale, max=1.0)
        for p in new_model.parameters():
            if p.grad is not None:
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)

    return new_model


def poisson_sampling(num_samples, sample_rate):
    indices = np.random.rand(num_samples) < sample_rate
    return indices
