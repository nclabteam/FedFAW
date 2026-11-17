import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import TensorDataset, DataLoader
from mak.utils.data import dataset_info
from mak.datasets.gender_classification import CelebAGenderDataset
# DomainNet / UCI-HAR / Capture-24 loaders are imported lazily in their branches
# below, so the code runs without those optional modules.

def process_domainnet(logger, total_clients, transform, test_batch_size):
    from mak.datasets.domainnet import DomainNetDataset, get_top_10_classes, DOMAINS
    domainnet_path = './datasets/domainnet'
    # For DomainNet Feature Skew, we expect 6 clients (one per domain)
    # However, for flexibility we check if it matches DOMAINS length
    if total_clients != len(DOMAINS):
        logger.warning(f"DomainNet feature skew experiment typically uses {len(DOMAINS)} clients (one per domain). Current config: {total_clients}")
    
    top_10_classes = get_top_10_classes(domainnet_path)
    if not top_10_classes:
        raise RuntimeError("DomainNet dataset missing. Run `bash scripts/download_domainnet.sh`.")
        
    logger.info(f"Top 10 DomainNet classes extracted: {top_10_classes}")
    
    train_datasets = []
    test_datasets = []
    client_indices = []
    net_cls_counts = {}
    current_idx = 0
    
    for i, domain in enumerate(DOMAINS):
        domain_train = DomainNetDataset(root=domainnet_path, domain=domain, train=True, 
                                        transform=transform, top_10_classes=top_10_classes)
        domain_test = DomainNetDataset(root=domainnet_path, domain=domain, train=False, 
                                       transform=transform, top_10_classes=top_10_classes)
        
        # Client indices belong to this specific domain mapping
        indices = list(range(current_idx, current_idx + len(domain_train)))
        client_indices.append(indices)
        current_idx += len(domain_train)
        
        train_datasets.append(domain_train)
        test_datasets.append(domain_test)
        
        # Calculate class counts for this client
        targets = [t for _, t in domain_train.samples]
        unq, unq_cnt = np.unique(targets, return_counts=True)
        net_cls_counts[i] = {int(k): int(v) for k, v in zip(unq, unq_cnt)}
        
    # Combine everything for the framework
    full_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    
    # Attach domain metadata to test_dataset so main.py can recover it
    test_dataset.domain_datasets = {i: ds for i, ds in enumerate(test_datasets)}
    test_dataset.domain_names = {i: name for i, name in enumerate(DOMAINS)}

    logger.info(f"DomainNet data partitioned natively by {len(DOMAINS)} domains representing Feature Skew.")
    return train_datasets, full_train_dataset, test_dataset, net_cls_counts, client_indices


def __getDirichletData__(y, n, alpha, num_c, min_samples_per_class=0):
    """
    min_samples_per_class > 0 guarantees every client gets at least that many
    examples of every class (floor, taken off the top before the Dirichlet
    split), on top of the usual alpha-controlled skew for the rest. Without
    this, a low alpha with many clients can starve a client of entire classes
    -- e.g. a client with only ~4/10 classes and one class at ~90% of its
    shard produced a near-singular per-layer curvature matrix that crashed
    FedBEns's KFAC Laplace eigendecomposition (CUDA illegal memory access).
    The floor is deliberately small so alpha still dominates the skew.
    """

    min_size = 0
    N = len(y)
    net_dataidx_map = {}
    p_client = np.zeros((n, num_c))

    for i in range(n):
        p_client[i] = np.random.dirichlet(np.repeat(alpha, num_c))
    idx_batch = [[] for _ in range(n)]

    for k in range(num_c):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)

        if min_samples_per_class > 0 and len(idx_k) > 0:
            floor_total = min(n * min_samples_per_class, len(idx_k))
            floor_idx, idx_k = idx_k[:floor_total], idx_k[floor_total:]
            for j, split in enumerate(np.array_split(floor_idx, n)):
                idx_batch[j] += split.tolist()

        proportions = p_client[:, k]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

    for j in range(n):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    local_sizes = []
    for i in range(n):
        local_sizes.append(len(net_dataidx_map[i]))
    local_sizes = np.array(local_sizes)
    weights = local_sizes / np.sum(local_sizes)

    print('Data statistics: %s' % str(net_cls_counts))
    print('Data ratio: %s' % str(weights))

    return idx_batch, net_cls_counts


def __getIIDData__(y, n):
    N = len(y)
    idx = np.arange(N)
    np.random.shuffle(idx)
    idx_batch = np.array_split(idx, n)

    net_cls_counts = {}
    for net_i, dataidx in enumerate(idx_batch):
        unq, unq_cnt = np.unique(y[dataidx], return_counts=True)
        net_cls_counts[net_i] = {unq[i]: unq_cnt[i] for i in range(len(unq))}

    local_sizes = np.array([len(batch) for batch in idx_batch])
    weights = local_sizes / np.sum(local_sizes)

    print('Data statistics: %s' % str(net_cls_counts))
    print('Data ratio: %s' % str(weights))

    return [batch.tolist() for batch in idx_batch], net_cls_counts

def get_dataset(logger, datatype, n_client, n_c, alpha, partition_equal=True, partition_scheme='dirichlet',
                min_samples_per_class=0):
    dataset_train = []
    dataset_train_global = None
    dataset_test_global = None
    net_cls_counts = None
    inds = None
    
    datatype = datatype.lower()
    if datatype not in dataset_info:
        # Fallback for data_fedfisher specifically if there are aliases
        if datatype == 'fashionmnist':
            datatype = 'fashion_mnist'
        elif datatype == 'cinic10':
             pass # already in dataset_info as per previous check
        else:
            raise ValueError(f"Unsupported dataset: {datatype}")

    if partition_scheme not in ('dirichlet', 'iid', 'natural'):
        raise ValueError(
            f"Unknown partition_scheme: {partition_scheme!r} (expected 'dirichlet', 'iid' or 'natural')")
    if partition_scheme == 'natural' and datatype not in ('uci_har', 'capture24'):
        raise ValueError("partition_scheme 'natural' is only supported for datasets uci_har and capture24")

    dataset_cfg = dataset_info[datatype]
    
    # Generic transforms based on dataset_info
    transform_list = [transforms.ToTensor()]
    if 'img_size' in dataset_cfg:
        transform_list.insert(0, transforms.Resize(dataset_cfg['img_size']))
    if datatype == 'celeba':
       transform_list.insert(1, transforms.CenterCrop(dataset_cfg['img_size'][0]))
    if 'mean' in dataset_cfg and 'std' in dataset_cfg:
        transform_list.append(transforms.Normalize(mean=dataset_cfg['mean'], std=dataset_cfg['std']))
    
    transform = transforms.Compose(transform_list)
    
    # Load datasets
    if datatype == 'cifar10':
        dataset_train_global = datasets.CIFAR10('./datasets', train=True, download=True, transform=transform)
        dataset_test_global = datasets.CIFAR10('./datasets', train=False, download=True, transform=transform)
    elif datatype == 'cifar100':
        dataset_train_global = datasets.CIFAR100('./datasets', train=True, download=True, transform=transform)
        dataset_test_global = datasets.CIFAR100('./datasets', train=False, download=True, transform=transform)
    elif datatype == 'mnist':
        dataset_train_global = datasets.MNIST('./datasets', train=True, download=True, transform=transform)
        dataset_test_global = datasets.MNIST('./datasets', train=False, download=True, transform=transform)
    elif datatype == 'fashion_mnist' or datatype == 'fashionmnist':
        dataset_train_global = datasets.FashionMNIST('./datasets', train=True, download=True, transform=transform)
        dataset_test_global = datasets.FashionMNIST('./datasets', train=False, download=True, transform=transform)
    elif datatype == 'svhn':
        dataset_train_global = datasets.SVHN('./datasets/svhn', split="train", download=True, transform=transform)
        dataset_test_global = datasets.SVHN('./datasets/svhn', split="test", download=True, transform=transform)
    elif datatype == 'stl':
        dataset_train_global = datasets.STL10('./datasets/stl10', split="train", download=True, transform=transform)
        dataset_test_global = datasets.STL10('./datasets/stl10', split="test", download=True, transform=transform)
    elif datatype == 'gtsrb':
        dataset_train_global = datasets.GTSRB('./datasets/gtsrb', split="train", download=True, transform=transform)
        dataset_test_global = datasets.GTSRB('./datasets/gtsrb', split="test", download=True, transform=transform)
    elif datatype == 'cinic10':
        dataset_train_global = datasets.ImageFolder('./datasets/cinic-10/train', transform=transform)
        dataset_test_global = datasets.ImageFolder('./datasets/cinic-10/test', transform=transform)
    elif datatype == 'celeba':
        dataset_train_global = CelebAGenderDataset(root="./datasets", split="train", transform=transform, download=True)
        dataset_test_global = CelebAGenderDataset(root="./datasets", split="test", transform=transform, download=True)
    elif datatype == 'domainnet':
        return process_domainnet(logger, n_client, transform, 32)
    elif datatype == 'uci_har':
        from mak.datasets.uci_har import load_uci_har, natural_subject_partition
        dataset_train_global, dataset_test_global, subject_train, y_train, _ = load_uci_har()

        if partition_scheme == 'natural':
            # One client per real UCI-HAR subject (30 total), instead of a synthetic split.
            inds, net_cls_counts = natural_subject_partition(subject_train, y_train)
            if n_client != len(inds):
                raise ValueError(
                    f"uci_har natural partitioning uses one client per real subject "
                    f"({len(inds)} subjects); set server.num_clients to {len(inds)} (got {n_client})."
                )
            for i, idx in enumerate(inds):
                print("Client ", i, " Training examples: ", len(idx))
            dataset_train = [Subset(dataset_train_global, idx) for idx in inds]
            return dataset_train, dataset_train_global, dataset_test_global, net_cls_counts, inds
        # 'dirichlet'/'iid' fall through to the generic partitioning block below,
        # the same synthetic split used for the image datasets above.
    elif datatype == 'capture24':
        from mak.datasets.capture24 import load_capture24, natural_participant_partition
        dataset_train_global, dataset_test_global, pid_train, y_train, _, classes = load_capture24('Willetts2018')
        if len(classes) != n_c:
            raise ValueError(f"capture24: label scheme has {len(classes)} classes but dataset_info says {n_c}")
        logger.info(f"capture24 classes: {classes} | train windows {len(dataset_train_global)}, "
                    f"test windows {len(dataset_test_global)}")

        if partition_scheme == 'natural':
            # Each client = all windows of a disjoint group of real training participants.
            inds, net_cls_counts, groups = natural_participant_partition(pid_train, y_train, n_client)
            for i, (idx, g) in enumerate(zip(inds, groups)):
                logger.info(f"Client {i}: participants {g} | {len(idx)} windows")
            dataset_train = [Subset(dataset_train_global, idx) for idx in inds]
            return dataset_train, dataset_train_global, dataset_test_global, net_cls_counts, inds
        # 'dirichlet'/'iid' fall through to the generic partitioning block below.

    if dataset_train_global is not None:
        train_loader = DataLoader(dataset_train_global, batch_size=len(dataset_train_global))
        # Use single batch to get all data
        data_iter = iter(train_loader)
        batch = next(data_iter)
        X_train = batch[0].numpy()
        Y_train = batch[1].numpy()

        if partition_scheme == 'iid':
            inds, net_cls_counts = __getIIDData__(Y_train, n_client)
        else:
            inds, net_cls_counts = __getDirichletData__(Y_train, n_client, alpha, n_c, min_samples_per_class)

        for i, ind in enumerate(inds):
            x = X_train[ind]
            y = Y_train[ind]
            x_train = torch.Tensor(x)
            y_train = torch.LongTensor(y)

            print ("Client ", i, " Training examples: " , len(x_train))
            dataset_train_torch = TensorDataset(x_train, y_train)
            dataset_train.append(dataset_train_torch)
    
    return dataset_train, dataset_train_global, dataset_test_global, net_cls_counts, inds
    