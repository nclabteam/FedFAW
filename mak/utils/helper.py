import os
import csv
import yaml
import json
import copy
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import date, datetime
from mak.clients.base_client import BaseClient
from torch.utils.data import DataLoader, Subset
from mak.servers.base_server import FedAvgServer
from mak.servers.scaffold_server import ScaffoldServer
from mak.clients.scaffold_client import ScaffoldClient
from mak.servers.fedfaw_nofisher_server import FedFAWNoFisherServer
from mak.clients.fisher_client import FisherClient
from mak.servers.fedfisher_server import FedFisherServer
from mak.servers.fedfaw_server import FedFAWServer
from mak.servers.fisher_merge_server import FisherMergeServer
from mak.servers.otfusion_server import OTFusionServer
from mak.servers.regmean_server import RegMeanServer
from mak.servers.pfnm_server import PFNMServer
from mak.servers.dense_server import DENSEServer
from mak.servers.fedlpa_server import FedLPAServer
from mak.clients.fedlpa_client import FedLPAClient
from mak.servers.fedbens_server import FedBEnsServer
from mak.clients.fedbens_client import FedBEnsClient
from mak.utils.comm_utils import get_size_mb
from mak.utils.data import evaluate_model
import json

from mak.models import *

def evaluate_domainnet_clients(selected_clients, test_loader, device, logger):
    local_accs = {}
    for c_id, client_obj in selected_clients.items():
        c_model = client_obj.model
        c_domain_loader = test_loader.domain_loaders.get(c_id)
        c_domain_name = test_loader.domain_names.get(c_id, f"Domain_{c_id}")
        
        if c_domain_loader:
            c_acc, _ = evaluate_model(c_model, c_domain_loader, device=device)
            short_key = c_domain_name[0] if isinstance(c_domain_name, str) else str(c_id)
            local_accs[short_key] = round(c_acc, 2)
            logger.info(f"Client {c_id} (Domain: {c_domain_name}) Local Post-Training Acc: {c_acc:.2f}%")
            
    return json.dumps(local_accs)


def client_selection(clients_dict, cpr, total_clients):
    # Calculate the number of clients to select
    num_clients_to_select = max(1, int(cpr * total_clients))

    # Get a list of client IDs
    client_ids = list(clients_dict.keys())

    # Randomly sample client IDs without replacement
    selected_ids = random.sample(client_ids, num_clients_to_select)

    # Create a new dictionary with the selected clients
    selected_clients = {
        client_id: clients_dict[client_id] for client_id in selected_ids}

    return selected_clients


def get_client(client_id, strategy, train_data, config, logger, model, device, out_dir, num_classes,test_loader=None):
    if strategy in ['fedavg', 'fedprox', 'otfusion', 'regmean', 'pfnm', 'dense', 'fedfaw_nofisher']:
        return BaseClient(client_id=client_id,
                          strategy=strategy, train_data=train_data,
                          logger=logger, config=config, model=model, device=device, out_dir=out_dir, num_classes=num_classes,test_loader=test_loader)
    elif strategy in ['scaffold']:
        return ScaffoldClient(client_id=client_id,
                              strategy=strategy, train_data=train_data,
                              logger=logger, config=config, model=model, device=device, out_dir=out_dir, num_classes=num_classes,test_loader=test_loader)
    elif strategy in ['fedfisher', 'fedfaw', 'fisher_merge']:
        return FisherClient(client_id=client_id,
                            strategy=strategy, train_data=train_data,
                            logger=logger, config=config, model=model, device=device, out_dir=out_dir, num_classes=num_classes,test_loader=test_loader)
    elif strategy in ['fedlpa']:
        return FedLPAClient(client_id=client_id,
                            strategy=strategy, train_data=train_data,
                            logger=logger, config=config, model=model, device=device, out_dir=out_dir, num_classes=num_classes,test_loader=test_loader)
    elif strategy in ['fedbens']:
        return FedBEnsClient(client_id=client_id,
                            strategy=strategy, train_data=train_data,
                            logger=logger, config=config, model=model, device=device, out_dir=out_dir, num_classes=num_classes,test_loader=test_loader)
    else:
        raise ValueError(f"Client {strategy} not found")


def generate_clients(total_clients, strategy, config, logger, global_model, out_dir, num_classes, device, full_dataset=None, client_indices=None, client_datasets=None,test_loader=None):
    all_clients = {}
    for client_id in tqdm(range(total_clients), desc='Generating Clients'):
        # Load client's data
        if client_datasets is not None:
            train_data = client_datasets[client_id]
        else:
            train_data = Subset(full_dataset, client_indices[client_id])
        # generate client object
        client = get_client(client_id=client_id,
                            strategy=strategy, train_data=train_data, config=config, logger=logger,
                            model=copy.deepcopy(global_model), device=device, out_dir=out_dir, num_classes=num_classes,test_loader=test_loader)
        all_clients[client_id] = client
    return all_clients


def get_device(config):

    device_id = config['common']['device']
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    if device_id == -1 or not torch.cuda.is_available():
        return torch.device("cpu")

    num_gpus = torch.cuda.device_count()

    if isinstance(device_id, int) and 0 <= device_id < num_gpus:
        return torch.device(f"cuda:{device_id}")
    else:
        raise ValueError(
            f"Invalid GPU ID: {device_id}. Available GPUs: {list(range(num_gpus))}")


def setup(args):

    config = get_config(args.config)

    seed = args.seed if args.seed else config.get('common', {}).get('seed', 42)
    device = args.gpu if args.gpu >= 0 else config.get(
        'common', {}).get('device', -1)
    strategy = args.strategy if args.strategy else config.get(
        'server', {}).get('strategy', '')

    dir_alpha = args.alpha if args.alpha else config.get(
        'common', {}).get('dirichlet_alpha', '')
    if args.noise != None:
        noise = True
        noise_sigma = args.noise
    else:
        noise = False
        noise_sigma = 0.0

    config['common']['seed'] = seed
    config['common']['dirichlet_alpha'] = dir_alpha
    config['common']['device'] = device
    config['server']['strategy'] = strategy
    config['noise']['enable'] = noise
    config['noise']['sigma'] = noise_sigma

    if args.fisher_type:
        config['fisher']['fisher_type'] = args.fisher_type

    if args.num_val_samples is not None:
        if 'fisher' not in config:
            config['fisher'] = {}
        config['fisher']['val_samples'] = args.num_val_samples

    if args.dataset:
        config['common']['dataset'] = args.dataset

    if args.model:
        config['common']['model'] = args.model

    if args.num_clients is not None:
        config['server']['num_clients'] = args.num_clients

    if args.base_dir:
        config['common']['base_dir'] = args.base_dir

    if args.epochs:
        config['client']['epochs'] = args.epochs
    if args.pretrained:
        config['common']['pretrained'] = args.pretrained

    # Overwrite FedFAW configs if provided from command line
    if 'fedfaw' not in config:
        config['fedfaw'] = {}
    
    if args.init_points:
        config['fedfaw']['init_points'] = args.init_points
    if args.n_iter:
        config['fedfaw']['n_iter'] = args.n_iter
    if args.acq:
        config['fedfaw']['acquisition_fn'] = args.acq
    if args.kappa:
        config['fedfaw']['kappa'] = args.kappa
    if args.xi:
        config['fedfaw']['xi'] = args.xi

    return config




def set_seed(logger, seed: int = 13):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set the seed for CUDA operations (if using GPU)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    logger.info(f"All random seeds set to {seed}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedFAW: one-shot federated learning")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/base.yaml",
        help="path to the config.yaml file."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="FL Strategy/algorithm"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for randomness"
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="add dp noise to data or not"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="-1 for cpu otherwise provide index of gpu"
    )
    parser.add_argument(
        "--ipc",
        type=int,
        default=None,
        help="IPC images per class for data condensation"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Dir(alpha) for Dirchilet distribution "
    )
    parser.add_argument(
        "--fisher_type",
        type=str,
        default=None,
        help="Fisher type: diagonal or kfac"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Total number of clients (overrides server.num_clients)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Output base directory (overrides common.base_dir)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name"
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="Use pretrained model"
    )

    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=None,
        help="Number of server validation samples (overrides config)"
    )

    # FedFAW specific arguments
    parser.add_argument(
        "--init_points",
        type=int,
        default=None,
        help="FedFAW: Number of random points to start search"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=None,
        help="FedFAW: Number of Bayesian optimization steps"
    )
    parser.add_argument(
        "--acq",
        type=str,
        default=None,
        help="FedFAW: Acquisition function (ei, poi, ucb)"
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="FedFAW: Kappa for UCB"
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=None,
        help="FedFAW: Xi for EI/POI"
    )

    args = parser.parse_args()


    return args


def get_config(file_path):
    # Open the YAML file
    with open(file_path, 'r') as file:
        # Parse the YAML data
        config = yaml.safe_load(file)
        return config


def gen_dir_outfile_server(config):
    # generates the basic directory structure for out data and the header for file
    today = date.today()
    BASE_DIR = config['common']['base_dir']
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    # create a date wise folder
    if not os.path.exists(os.path.join(BASE_DIR, str(today))):
        os.mkdir(os.path.join(BASE_DIR, str(today)))

    # create saperate folder based on strategy
    if not os.path.exists(os.path.join(BASE_DIR, str(today), config['server']['strategy'])):
        os.mkdir(os.path.join(BASE_DIR, str(today),
                 config['server']['strategy']))

    dirs = os.listdir(os.path.join(BASE_DIR, str(today),
                                   config['server']['strategy']))
    final_dir_path = os.path.join(BASE_DIR, str(
        today), config['server']['strategy'], str(len(dirs)))

    if not os.path.exists(final_dir_path):
        os.mkdir(final_dir_path)
        os.mkdir(os.path.join(final_dir_path, 'models'))

    if config['server']['strategy'] in ['fedaf', 'feddatadist']:
        os.mkdir(os.path.join(final_dir_path, 'images'))
        os.mkdir(os.path.join(final_dir_path, 'logits'))
        os.mkdir(os.path.join(final_dir_path, 'results'))

    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    # save all confugration file as json file
    json_file_name = f"config.json"
    with open(os.path.join(final_dir_path, json_file_name), 'w') as fp:
        json.dump(config, fp, indent=4)
    dataset_str = config['common']['dataset'].replace('/', '_')
    file_name = f"{config['server']['strategy']}_{dataset_str}_{config['client']['test_batch_size']}_{config['client']['lr']}_{config['client']['epochs']}"
    file_name = f"{file_name}.csv"
    out_file_path = os.path.join(
        final_dir_path, file_name)
    # create empty server history file
    if not os.path.exists(out_file_path):
        with open(out_file_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            header = ["round", "accuracy", "loss",
                      "time", "uploaded_mib", "downloaded_mib", "domain_accs"]
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()
    return out_file_path, final_dir_path


def set_logger(log_file_path: str) -> logging.Logger:
    """
    Sets up a logger with both file and console handlers.

    Args:
        log_file_path (str): The full path to the log file.

    Returns:
        logging.Logger: Configured logger object.
    """
    log_file_path = log_file_path.replace('.csv', '.log')
    # Ensure the directory for the log file exists
    base_path = os.path.dirname(log_file_path)
    os.makedirs(base_path, exist_ok=True)

    # Configure logger
    logger = logging.getLogger(os.path.basename(log_file_path))
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Formatter for the log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - Line %(lineno)d - %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.info(f"Logging to : {log_file_path}")
    return logger


def update_results_csv(file_path, round, acc, loss, time, uploaded_mib, downloaded_mib,domain_accs=None):
    field_names = ["round", "accuracy", "loss",
                   "time", "uploaded_mib", "downloaded_mib","domain_accs"]
    dict = {"round": round, "accuracy": acc, "loss": loss,
            "time": time, "uploaded_mib": uploaded_mib, "downloaded_mib": downloaded_mib,"domain_accs":domain_accs}
    with open(file_path, 'a') as f:
        dictwriter_object = csv.DictWriter(f, fieldnames=field_names)
        dictwriter_object.writerow(dict)
        f.close()


def get_server_cls(strategy, model, device, config, logger, out_dir, validation_dataset=None, full_dataset=None, client_indices=None):
    if strategy in ['fedavg', 'fedprox']:
        return FedAvgServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset)
    elif strategy in ['scaffold']:
        return ScaffoldServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset)
    elif strategy in ['fedfisher']:
        return FedFisherServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset, full_dataset=full_dataset, client_indices=client_indices)
    elif strategy in ['fedfaw']:
        return FedFAWServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset, full_dataset=full_dataset, client_indices=client_indices)
    elif strategy in ['fisher_merge']:
        return FisherMergeServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset, full_dataset=full_dataset, client_indices=client_indices)
    elif strategy in ['otfusion']:
        return OTFusionServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset)
    elif strategy in ['regmean']:
        return RegMeanServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset, full_dataset=full_dataset, client_indices=client_indices)
    elif strategy in ['pfnm']:
        return PFNMServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset, full_dataset=full_dataset, client_indices=client_indices)
    elif strategy in ['dense']:
        return DENSEServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset)
    elif strategy in ['fedlpa']:
        return FedLPAServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset, full_dataset=full_dataset, client_indices=client_indices)
    elif strategy in ['fedbens']:
        return FedBEnsServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset)
    elif strategy in ['fedfaw_nofisher']:
        return FedFAWNoFisherServer(config=config, model=copy.deepcopy(model), logger=logger, device=device, out_dir=out_dir, validation_dataset=validation_dataset, full_dataset=full_dataset, client_indices=client_indices)
    else:
        raise ValueError('Unknown Server: %s' % strategy)
