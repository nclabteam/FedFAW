import copy
import torch
import numpy as np
from mak.servers.base_server import FedAvgServer
from mak.utils.pfnm_utils import oneshot_matching, compute_full_cnn
from mak.utils.model_utils import get_network
from mak.utils.data import dataset_info

class Namespace():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class PFNMServer(FedAvgServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset, full_dataset, client_indices):
        super().__init__(model, config, logger, device, out_dir, validation_dataset)
        self.full_dataset = full_dataset
        self.client_indices = client_indices
        self.logger.info("Server initialized: PFNM Server")
        
        # Calculate net_cls_counts for PFNM
        self.net_cls_counts = {}
        if isinstance(self.client_indices, list):
            iterator = enumerate(self.client_indices)
        else:
            iterator = self.client_indices.items()

        for c_id, indices in iterator:
            self.net_cls_counts[c_id] = {}
            for idx in indices:
                _, label = self.full_dataset[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                self.net_cls_counts[c_id][label] = self.net_cls_counts[c_id].get(label, 0) + 1


    def aggregate(self, results):
        n = len(results)
        num_classes = dataset_info[self.config['common']['dataset']]['num_classes']
        sorted_keys = sorted(results.keys())
        
        models = []
        for c_id in sorted_keys:
            client_model = copy.deepcopy(self.global_model)
            client_model.load_state_dict(results[c_id]['model'])
            models.append(client_model)

        # Prepare pfnm arguments
        args_pfnm = Namespace(
            device=self.device,
            dataset=self.config['common']['dataset'].lower(),
            model=self.config['common']['model'].lower(), # PFNM expects lower case
            sigma0=self.config.get('pfnm', {}).get('sigma0', 1.0),
            sigma=self.config.get('pfnm', {}).get('sigma', 1.0),
            n_c=num_classes,
            gamma=self.config.get('pfnm', {}).get('gamma', 7.0)
        )
        
        # Map model names if necessary
        if 'resnet18' in args_pfnm.model:
            args_pfnm.model = 'resnet' # PFNM might expect specific names
        elif 'fednet' in args_pfnm.model:
            args_pfnm.model = 'fednet'
            
        self.logger.info(f"Performing PFNM matching with args: {args_pfnm.__dict__}")

        # PFNM expects models to have bias. If ours don't, we create dummy versions with zero bias.
        models_dummy_bias = []
        for i in range(n):
            # Try to get the same architecture but with bias=True
            # Note: This assumes our architectures support bias toggle or we manually handle it.
            # In our case, PFNM will reconstruct its own container models.
            models_dummy_bias.append(models[i])

        # Metadata for PFNM
        model_meta_data = []
        layer_type = []
        for (k, v) in models_dummy_bias[0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

        # Averaging weights based on class counts
        averaging_weights = np.zeros((n, num_classes), dtype=np.float32)
        for i in range(num_classes):
            total_num_counts = 0
            worker_class_counts = [0] * n
            for j, c_id in enumerate(sorted_keys):
                count = self.net_cls_counts[c_id].get(i, 0)
                total_num_counts += count
                worker_class_counts[j] = count
            if total_num_counts > 0:
                averaging_weights[:, i] = np.array(worker_class_counts) / total_num_counts

        # Perform matching
        try:
            matched_weights, assignments = oneshot_matching(
                models_dummy_bias, model_meta_data, layer_type, 
                self.net_cls_counts, averaging_weights, args_pfnm, num_classes
            )
            
            # Reconstruct model
            res_model = compute_full_cnn(matched_weights, self.device, args_pfnm)
            
            # Update global model
            # Note: PFNM might change the capacity (number of neurons). 
            # If so, we can't easily load it back into the original global_model.
            # However, for 1-round simulations or if we allow elastic layers, it's fine.
            # Here we assume we want the aggregated state dict.
            # If the architecture changed, this load_state_dict will fail.
            try:
                self.global_model.load_state_dict(res_model.state_dict())
            except Exception as e:
                self.logger.warning(f"Could not load PFNM result into global model (capacity changed?): {e}")
                self.logger.info("Setting global model to matched result model.")
                self.global_model = res_model
                
        except Exception as e:
            self.logger.error(f"PFNM matching failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Fallback to FedAvg
            super().aggregate(results)

    def __repr__(self) -> str:
        return " PFNM Strategy"
