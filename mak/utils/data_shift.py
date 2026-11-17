import torch
import numpy as np
from mak.utils.data_fedfisher import get_dataset
from mak.utils.data import dataset_info

def apply_val_data_shift(config, full_dataset, num_classes, logger):
    """
    Applies server-side validation data shift (label, domain, or covariate)
    based on the configuration dictionary.
    """
    shift_cfg = config.get('val_data_shift', {})
    shift_type = shift_cfg.get('shift_type', 'None')
    val_samples = config['fisher']['val_samples']

    if shift_type != 'None' and shift_type is not None:
        logger.info(f"Applying server-side val_data_shift: {shift_type}")
        if shift_type == 'label':
            alpha_shift = shift_cfg.get('label_shift_dir_alpha', 0.1)
            logger.info(f"Label shift Dirichlet alpha: {alpha_shift}")

            if hasattr(full_dataset, 'targets'):
                targets = np.array(full_dataset.targets)
            else:
                logger.info("Extracting label targets explicitly...")
                targets = [full_dataset[i][1] for i in range(len(full_dataset))]
                if isinstance(targets[0], torch.Tensor):
                    targets = [t.item() for t in targets]
                targets = np.array(targets)
                
            p_val = np.random.dirichlet(np.repeat(alpha_shift, num_classes))
            val_counts = (p_val * val_samples).astype(int)
            while val_counts.sum() < val_samples:
                val_counts[np.random.randint(num_classes)] += 1
            while val_counts.sum() > val_samples:
                idx_to_decrease = np.random.choice(np.where(val_counts > 0)[0])
                val_counts[idx_to_decrease] -= 1
                
            ind = []
            for k in range(num_classes):
                idx_k = np.where(targets == k)[0]
                np.random.shuffle(idx_k)
                ind.extend(idx_k[:val_counts[k]].tolist())
            
            dataset_val = torch.utils.data.Subset(full_dataset, ind)
            
        elif shift_type == 'domain':
            domain_ds_name = shift_cfg.get('domain_shift_dataset')
            logger.info(f"Fetching shift domain target: {domain_ds_name}")
            # Use the test split of the shift-domain dataset
            _, _, domain_full, _, _ = get_dataset(logger=logger, datatype=domain_ds_name, n_client=1, n_c=num_classes, alpha=100.0)
            ind = np.random.choice(len(domain_full), min(val_samples, len(domain_full)), replace=False)
            dataset_val = torch.utils.data.Subset(domain_full, ind)
            # Samples come from a different dataset, so nothing to exclude from the test set
            return dataset_val, None
            
        elif shift_type == 'covariate':
            cov_type = shift_cfg.get('covariate_shift_type')
            cov_strength = shift_cfg.get('covariate_shift_strength', 0.1)
            logger.info(f"Covariate shift type: {cov_type}, strength: {cov_strength}")
            ind = np.random.choice(len(full_dataset), min(val_samples, len(full_dataset)), replace=False)

            # full_dataset's own transform already applies Normalize(mean, std),
            # so images arriving in __getitem__ below are NOT raw [0,1] pixels.
            # torchvision's adjust_brightness/adjust_contrast internally clamp
            # float tensors to [0,1] (see torchvision._functional_tensor._blend)
            # -- applying them directly to an already-normalized tensor (which
            # routinely has values <0 or >1) clamps most of the image to the
            # two extremes instead of adjusting brightness/contrast. Undo the
            # normalization first, adjust in real pixel space, then re-apply it.
            ds_cfg = dataset_info[config['common']['dataset']]
            mean_t = torch.tensor(ds_cfg['mean']).view(-1, 1, 1)
            std_t = torch.tensor(ds_cfg['std']).view(-1, 1, 1)

            class CovariateShiftSubset(torch.utils.data.Dataset):
                def __init__(self, subset, cov_type, cov_strength, mean, std):
                    self.subset = subset
                    self.cov_type = cov_type
                    self.cov_strength = cov_strength
                    self.mean = mean
                    self.std = std
                    import torchvision.transforms.functional as TF
                    self.TF = TF

                def __getitem__(self, idx):
                    x, y = self.subset[idx]
                    if self.cov_type in ('brightness', 'contrast'):
                        mean = self.mean.to(x.device, x.dtype)
                        std = self.std.to(x.device, x.dtype)
                        x_pixel = (x * std + mean).clamp(0.0, 1.0)
                        if self.cov_type == 'brightness':
                            x_pixel = self.TF.adjust_brightness(x_pixel, 1.0 + self.cov_strength)
                        else:
                            x_pixel = self.TF.adjust_contrast(x_pixel, 1.0 + self.cov_strength)
                        x = (x_pixel - mean) / std
                    elif self.cov_type == 'noise':
                        noise = torch.randn_like(x) * self.cov_strength
                        x = x + noise
                    return x, y

                def __len__(self):
                    return len(self.subset)

            base_dataset_val = torch.utils.data.Subset(full_dataset, ind)
            dataset_val = CovariateShiftSubset(base_dataset_val, cov_type, cov_strength, mean_t, std_t)
            
        else:
            raise ValueError(f"Unsupported val_data_shift type: {shift_type}")
    else:
        logger.info("No data shift applied")
        ind = np.random.choice(len(full_dataset), min(val_samples, len(full_dataset)), replace=False)
        dataset_val = torch.utils.data.Subset(full_dataset, ind)

    # Indices (into the dataset passed in) used for the val set, so the caller
    # can exclude them from evaluation and keep val/test disjoint.
    return dataset_val, np.asarray(ind, dtype=int)


def exclude_indices(dataset, indices):
    """
    Return a Subset of `dataset` without `indices`. If `dataset` carries
    DomainNet-style per-domain metadata (a ConcatDataset with
    `domain_datasets` / `domain_names`), the per-domain datasets are filtered
    the same way so per-domain evaluation also stays disjoint from val.
    """
    if indices is None or len(indices) == 0:
        return dataset
    mask = np.ones(len(dataset), dtype=bool)
    mask[np.asarray(indices, dtype=int)] = False
    remaining = torch.utils.data.Subset(dataset, np.where(mask)[0].tolist())

    if hasattr(dataset, 'domain_datasets') and hasattr(dataset, 'cumulative_sizes'):
        starts = [0] + list(dataset.cumulative_sizes[:-1])
        remaining.domain_datasets = {
            i: torch.utils.data.Subset(ds, np.where(mask[starts[i]:starts[i] + len(ds)])[0].tolist())
            for i, ds in dataset.domain_datasets.items()
        }
        remaining.domain_names = dataset.domain_names
    return remaining


def calculate_shift_metrics(config, dataset_val, client_datasets, global_model, num_classes, logger, out_dir):
    import os
    import json
    import torch
    from scipy.stats import spearmanr
    from torchmetrics.image.fid import FrechetInceptionDistance

    logger.info("Calculating data shift metrics (Spearman rho and Frechet Inception Distance)...")

    # 1. Spearman Rho (Label Rank Alignment)
    val_counts = np.zeros(num_classes)
    for i in range(len(dataset_val)):
        _, y = dataset_val[i]
        val_counts[int(y)] += 1

    rho_list = []
    for c_id, client_ds in enumerate(client_datasets):
        c_counts = np.zeros(num_classes)
        for i in range(len(client_ds)):
            _, y = client_ds[i]
            c_counts[int(y)] += 1
            
        rho, _ = spearmanr(val_counts, c_counts)
        # NaN happens only if all elements are exactly identical uniformly across classes
        if np.isnan(rho):
            rho = 0.0
        rho_list.append(float(rho))
        
    avg_rho = float(np.mean(rho_list))
    logger.info(f"Average Spearman Rho across {len(client_datasets)} clients: {avg_rho:.4f}")

    # 2. Frechet Inception Distance (FID) -> Measures Covariate/Domain Feature Shift
    device = next(global_model.parameters()).device

    # Each side must be un-normalized with the stats its own transform used:
    # clients always use the training dataset's, while a domain-shifted val
    # set was loaded (and normalized) with the shift dataset's own stats.
    shift_cfg = config.get('val_data_shift', {})
    train_ds_name = config['common']['dataset']
    val_ds_name = (shift_cfg.get('domain_shift_dataset', train_ds_name)
                   if shift_cfg.get('shift_type') == 'domain' else train_ds_name)

    def get_norm_stats(ds_name):
        ds_cfg = dataset_info.get(ds_name, {})
        if 'mean' not in ds_cfg or 'std' not in ds_cfg:
            logger.warning(f"No mean/std for '{ds_name}'; assuming FID inputs are already in [0, 1].")
            return None, None
        mean = torch.tensor(ds_cfg['mean'], device=device).view(1, -1, 1, 1)
        std = torch.tensor(ds_cfg['std'], device=device).view(1, -1, 1, 1)
        return mean, std

    def process_and_add_batch(fid_metric, dataset, mean, std, is_real=True, max_samples=500):
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        count = 0
        for x, _ in loader:
            if count >= max_samples:
                break

            x = x.to(device)
            # Undo Normalize(mean, std) back to real [0, 1] pixel space. A
            # per-image min-max rescale would instead stretch every image to
            # the full range, erasing exactly the global intensity changes
            # (e.g. brightness covariate shift) FID is meant to detect.
            if mean is not None:
                x = x * std + mean
            x = x.clamp(0.0, 1.0)

            # Ensure strictly 3 RGB channels (e.g. padding MNIST)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            x_uint8 = (x * 255).round().to(torch.uint8)

            fid_metric.update(x_uint8, real=is_real)
            count += x.shape[0]

    val_mean, val_std = get_norm_stats(val_ds_name)
    train_mean, train_std = get_norm_stats(train_ds_name)

    fid_distances = []
    # Compute up to 5 clients to save upfront compute time (Inception is heavy)
    eval_clients = min(len(client_datasets), 5)
    for c_id in range(eval_clients):
        logger.info(f"Computing FID representation distance for Client {c_id}...")
        
        # Instantiate a fresh metric to ensure internal features memory blocks don't collide
        # feature=64 calculates much quicker than 2048 and maintains valid relative distances!
        fid = FrechetInceptionDistance(feature=64).to(device)
        
        process_and_add_batch(fid, dataset_val, val_mean, val_std, is_real=True)
        process_and_add_batch(fid, client_datasets[c_id], train_mean, train_std, is_real=False)
        
        try:
            score = float(fid.compute().item())
        except Exception as e:
            logger.warning(f"FID computation fundamentally failed for Client {c_id}: {e}")
            score = 0.0
            
        fid_distances.append(score)
        
    avg_fid_dist = float(np.mean(fid_distances))
    logger.info(f"Average Frechet Inception Distance (Subset of {eval_clients} clients): {avg_fid_dist:.4f}")
    
    # Save the metrics to a json file
    metrics = {
        "avg_spearman_rho": avg_rho,
        "per_client_rho": rho_list,
        "avg_fid_distance": avg_fid_dist,
        "per_client_fid_distance": fid_distances
    }
    
    out_file = os.path.join(out_dir, "data_shift_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Shift metrics dynamically derived and securely saved to {out_file}")
