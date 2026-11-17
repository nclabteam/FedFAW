from mak.utils.helper import parse_args, setup, gen_dir_outfile_server, set_logger
from mak.utils.helper import set_seed, get_device, generate_clients, get_server_cls
from mak.utils.model_utils import get_network
from mak.utils.helper import client_selection, update_results_csv
from mak.utils.data import dataset_info, load_data
from mak.utils.data_fedfisher import get_dataset
import time
import torch
import os
import numpy as np
from torchsummary import summary
from mak.utils.data_shift import apply_val_data_shift, exclude_indices
from mak.utils.helper import evaluate_domainnet_clients


supported_models = ['MLP', 'HARNet', 'ConvNet', 'LeNet', 'AlexNet',
                    'AlexNetBN', 'VGG11', 'VGG11BN', 'ResNet18', 'FedNet', 'FedAvgCNN']
supported_strategies = ['fedavg', 'fedprox', 'scaffold', 'otfusion', 'regmean', 'pfnm', 'dense', 'fisher_merge',
                        'fedlpa', 'fedfisher', 'fedbens', 'fedfaw', 'fedfaw_nofisher']


def simulate():
    # Parse arguments and configs
    args = parse_args()
    config = setup(args=args)

    # Generate directories and setup logger
    out_file_path, final_dir_path = gen_dir_outfile_server(config=config)
    logger = set_logger(log_file_path=out_file_path)

    strategy = config['server']['strategy']
    model_name = config['common']['model']
    total_clients = config['server']['num_clients']
    total_rounds = config['server']['num_rounds']
    cpr = config['server']['fraction_fit']

    if strategy not in supported_strategies:
        raise ValueError(f"Unsupported FL Strategy: {strategy}")
    if model_name not in supported_models:
        raise ValueError(f"Unsupported Model: {model_name}")
    if model_name == 'FedAvgCNN' and strategy in ('otfusion', 'pfnm'):
        raise ValueError(f"{strategy} does not support FedAvgCNN (OTFusion assumes bias-free layers; "
                         f"PFNM only supports its own fixed architectures). Use FedNet instead.")
    if config['common']['dataset'] in ('uci_har', 'capture24') and model_name not in ('MLP', 'HARNet'):
        raise ValueError(f"{config['common']['dataset']} is a tabular dataset and only supports model: MLP or HARNet")

    device = get_device(config)
    logger.info(f"Using Device : {device}")

    total_clients = config['server']['num_clients']
    total_rounds = config['server']['num_rounds']
    cpr = config['server']['fraction_fit']
    dataset_name = config['common']['dataset']
    num_classes = dataset_info[dataset_name]['num_classes']
    dir_alpha = config['common']['dirichlet_alpha']
    dataset_cfg = dataset_info[dataset_name]
    pretrained = config['common']['pretrained']
    if pretrained:
        config['client']['lr'] = 0.001
    
    # New data loading logic from data_fedfisher
    np.random.seed(3)
    
    partition_scheme = config['common'].get('partition_scheme', 'dirichlet')
    min_samples_per_class = config['common'].get('min_samples_per_class', 0)
    client_datasets, full_dataset, test_dataset, net_cls_counts, client_indices = get_dataset(logger=logger,
        datatype=dataset_name, n_client=total_clients, n_c=num_classes, alpha=dir_alpha,
        partition_scheme=partition_scheme, min_samples_per_class=min_samples_per_class)

    # Small validation set for fisher matrix calculation, sampled from the test split.
    # The sampled indices are then removed from the test set so val and test are disjoint.
    dataset_val, val_indices = apply_val_data_shift(config, test_dataset, num_classes, logger)
    test_dataset = exclude_indices(test_dataset, val_indices)
    logger.info(f"Server val samples: {len(dataset_val)} | held-out test samples: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['client']['test_batch_size'])
    
    # Transfer domain metadata if available (e.g., for DomainNet)
    if hasattr(test_dataset, 'domain_datasets'):
        test_loader.domain_loaders = {
            i: torch.utils.data.DataLoader(ds, batch_size=config['client']['test_batch_size'], shuffle=False)
            for i, ds in test_dataset.domain_datasets.items()
        }
        test_loader.domain_names = test_dataset.domain_names


    # Set seed for reproducibility
    seed = config['common']['seed']
    set_seed(seed=seed, logger=logger)


    global_model = get_network(model=model_name, channel=dataset_cfg.get('channel'), num_classes=dataset_cfg['num_classes'],
                               im_size=dataset_cfg.get('img_size'), device=device, seed=seed, use_pretrained=pretrained,
                               pretrained_path=config['common']['pretrained_path'], input_dim=dataset_cfg.get('input_dim'))

    all_clients = generate_clients(total_clients=total_clients,
                                   strategy=strategy, config=config, logger=logger, global_model=global_model,
                                   out_dir=final_dir_path, num_classes=num_classes, device=device,
                                   client_datasets=client_datasets,test_loader=test_loader)

    # calculate the weights of the local models proportional to datasize
    total_samples = sum(all_clients[c_id].data_size for c_id in all_clients)
    weights = [all_clients[c_id].data_size /
               total_samples for c_id in range(total_clients)]
    logger.info(f"Client Weights: {weights}")
    
    # Calculate Data Shift metrics mathematically and procedurally if requested
    if config.get('val_data_shift', {}).get('calculate_shift_metrics', False):
        from mak.utils.data_shift import calculate_shift_metrics
        calculate_shift_metrics(config, dataset_val, client_datasets, global_model, num_classes, logger, final_dir_path)

    # generate server object
    server = get_server_cls(config=config, strategy=strategy, model=global_model, device=device,
                            logger=logger, out_dir=final_dir_path, validation_dataset=dataset_val, full_dataset=full_dataset, client_indices=client_indices)

    # Main communication rounds
    test_acc = []
    test_loss = []
    total_time = 0
    for r in range(1, total_rounds + 1):
        logger.info(f"++++++ Starting Round {r}/{total_rounds} ++++++")
        round_start_time = time.time()
        # Distribute global model to clients
        global_model = server.get_global_model()

        # client selection
        if total_rounds == 1:
            selected_clients = all_clients
        else:
            selected_clients = client_selection(
                clients_dict=all_clients, cpr=cpr, total_clients=total_clients)
        logger.info(
            f"Selected {len(selected_clients)}/ {len(all_clients)} clients")

        upload_cost, download_cost = server.fit_round(
            clients=selected_clients, round=r)
        acc, loss = server.evaluate(round_num=r, test_loader=test_loader)
        test_acc.append(acc)
        test_loss.append(loss)

        domain_accs_str = ""
        if dataset_name == 'domainnet':
            domain_accs_str = evaluate_domainnet_clients(selected_clients, test_loader, device, logger)
            logger.info(f"Round {r} Domain Accuracies: {domain_accs_str}")

        c_round_time = time.time() - round_start_time
        acc_fl = test_acc[-1] if test_acc else 0
        loss_fl = test_loss[-1] if test_loss else 0
        update_results_csv(file_path=out_file_path, round=r,
                           acc=acc_fl, loss=loss_fl, time=c_round_time,
                           uploaded_mib=upload_cost, downloaded_mib=download_cost,domain_accs=domain_accs_str)
        logger.info(f"Round {r} finished in : {c_round_time}")
        total_time += c_round_time
    # save the final model
    if config['common']['save_model']:
        torch.save(server.get_global_model(), os.path.join(
            final_dir_path, 'models', f'model_round_{r}.pt'))
    logger.info(f"{total_rounds} FL Finished  in {total_time}")
    logger.info(f"Final Model Accuracies Seed {seed} : {test_acc}")


if __name__ == '__main__':
    simulate()
