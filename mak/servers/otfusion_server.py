import copy
import torch
from tqdm import tqdm
from mak.servers.base_server import FedAvgServer
from mak.utils.data import evaluate_model
from mak.utils.ot_utils import one_shot_otfusion
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

class OTFusionServer(FedAvgServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset):
        super().__init__(model, config, logger, device, out_dir, validation_dataset)
        self.logger.info("Server initialized: OTFusion Server")
        self.validation_loader = DataLoader(
            validation_dataset, batch_size=config['client']['test_batch_size'], shuffle=False
        ) if validation_dataset else None

    def aggregate(self, results):
        total_samples = sum(entry['num_samples'] for entry in results.values())
        models = []
        p = []
        
        # Sort results by key for consistency
        sorted_keys = sorted(results.keys())
        for c_id in sorted_keys:
            client_data = results[c_id]
            # Create model object from state dict
            client_model = copy.deepcopy(self.global_model)
            client_model.load_state_dict(client_data['model'])
            models.append(client_model)
            p.append(client_data['num_samples'] / total_samples)

        self.logger.info("Performing OTFusion aggregation...")
        
        updated_model = self.get_otfusion_model(models, p)
        
        # Update the global model parameters
        self.global_model.load_state_dict(updated_model.state_dict())
        self.logger.info("OTFusion aggregation completed.")

    def get_otfusion_model(self, models, p):
        # Default OTFusion arguments
        args_ot = {
            'eval_aligned': True,
            'gpu_id': self.device.index if self.device.type == 'cuda' else -1,
            'importance': None,
            'exact': True,
            'past_correction': False,
            'correction': False,
            'clip_gm': False,
            'not_squared': True,
            'proper_marginals': False,
            'ensemble_step': 0.5,
            'skip_last_layer': False,
            'handle_skips': False,
            'ground-metric': 'euclidean',
            'ground-metric-normalize': 'none',
            'reg': 1e-7,
            'dist_normalize': False,
            'activation_histograms': False,
            'debug': False,
            'geom-ensemble-type': 'wts',
            'ground-metric-eff': True,
            'normalize-wts': False,
            'unbalanced': False,
            'pivot': 0
        }

        # Override with config if present
        if 'otfusion' in self.config:
            args_ot.update(self.config['otfusion'])

        if self.config['common']['model'] == 'ResNet18':
            args_ot['handle_skips'] = True

        # Hyperparameter tuning on validation set
        correction_range = [True, False]
        past_correction_range = [True, False]
        best_acc = -1.0
        best_correction = False
        best_past_correction = False

        if self.validation_loader:
            self.logger.info("Tuning OTFusion hyperparameters (correction, past_correction) on validation set...")
            for correction in correction_range:
                for past_correction in past_correction_range:
                    args_ot['correction'] = correction
                    args_ot['past_correction'] = past_correction
                    
                    candidate_model = one_shot_otfusion(self.global_model, models, p, args_ot)
                    acc, _ = evaluate_model(candidate_model, self.validation_loader, self.device)
                    
                    self.logger.info(f"OTFusion Params - correction: {correction}, past_correction: {past_correction} => Acc: {acc:.4f}")
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_correction = correction
                        best_past_correction = past_correction
            
            self.logger.info(f"Best OTFusion Params selected - correction: {best_correction}, past_correction: {best_past_correction}")
            args_ot['correction'] = best_correction
            args_ot['past_correction'] = best_past_correction
        else:
            self.logger.warning("No validation dataset provided. Using default OTFusion parameters.")

        final_model = one_shot_otfusion(self.global_model, models, p, args_ot)
        return final_model

    def __repr__(self) -> str:
        return " OTFusion Strategy"
