import copy
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from mak.servers.base_server import FedAvgServer
from mak.utils.data import dataset_info
from mak.utils.dense_utils import Generator, AdvSynthesizer, Ensemble, KLDiv, kd_train

class DENSEServer(FedAvgServer):
    def __init__(self, model, config, logger, device, out_dir, validation_dataset):
        super().__init__(model, config, logger, device, out_dir, validation_dataset)
        self.logger.info("Server initialized: DENSE Server")
        
        # Generator settings
        self.nz = self.config.get('dense', {}).get('nz', 256)
        self.iterations = self.config.get('dense', {}).get('iterations', 30)
        self.lr_g = self.config.get('dense', {}).get('lr_g', 1e-3)
        self.synthesis_batch_size = self.config.get('dense', {}).get('synthesis_batch_size', 256)
        self.sample_batch_size = self.config.get('dense', {}).get('sample_batch_size', 256)
        self.adv = self.config.get('dense', {}).get('adv', 0.5)
        self.bn = self.config.get('dense', {}).get('bn', 1.0)
        self.oh = self.config.get('dense', {}).get('oh', 1.0)
        self.epochs = self.config.get('dense', {}).get('epochs', 20)
        self.T = self.config.get('dense', {}).get('T', 20)
        
        dataset_name = self.config['common']['dataset']
        nc = dataset_info[dataset_name]['channel']
        img_size = dataset_info[dataset_name]['img_size'][0]
        self.img_size_dims = (nc, img_size, img_size)
        
        self.generator = Generator(nz=self.nz, ngf=64, img_size=img_size, nc=nc).to(self.device)
        self.criterion = KLDiv(T=self.T)

    def aggregate(self, results):
        # 1. Standard FedAvg as a starting point
        total_samples = sum(entry['num_samples'] for entry in results.values())
        weighted_weights = None
        
        model_list = []
        p = []
        
        sorted_keys = sorted(results.keys())
        for c_id in sorted_keys:
            client_data = results[c_id]
            weight = client_data['num_samples'] / total_samples
            p.append(weight)
            
            client_model = copy.deepcopy(self.global_model)
            client_model.load_state_dict(client_data['model'])
            model_list.append(client_model)
            
            if weighted_weights is None:
                weighted_weights = {k: v * weight for k, v in client_data['model'].items()}
            else:
                for k, v in client_data['model'].items():
                    weighted_weights[k] += v * weight
                    
        # Official DENSE (zj-jayzhang/DENSE, loop_df_fl.py L281) distills into a
        # *freshly initialised* model (`global_model = get_model(args)`), not the
        # FedAvg model, and uses a uniform logit average (Ensemble_A). Both are
        # configurable; defaults follow the official code.
        student_init = self.config.get('dense', {}).get('student_init', 'random')
        if student_init == 'fedavg':
            self.global_model.load_state_dict(weighted_weights)
        else:
            for m in self.global_model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
        if self.config.get('dense', {}).get('ensemble_weighting', 'uniform') == 'uniform':
            p = [1.0 / len(model_list)] * len(model_list)
        self.logger.info(f"DENSE student init: {student_init}, ensemble weights: {p}. Starting distillation...")

        # 2. DENSE Distillation
        ensemble_model = Ensemble(model_list, p).to(self.device)
        
        save_dir = os.path.join(self.out_dir, "dense_pool")
        synthesizer = AdvSynthesizer(
            teacher=ensemble_model,
            student=self.global_model,
            generator=self.generator,
            nz=self.nz,
            num_classes=dataset_info[self.config['common']['dataset']]['num_classes'],
            img_size=self.img_size_dims,
            iterations=self.iterations,
            lr_g=self.lr_g,
            synthesis_batch_size=self.synthesis_batch_size,
            sample_batch_size=self.sample_batch_size,
            adv=self.adv,
            bn=self.bn,
            oh=self.oh,
            save_dir=save_dir,
            dataset=self.config['common']['dataset'].lower(),
            mean=self.mean,
            std=self.std
        )
        
        lr_dense = self.config.get('dense', {}).get('lr_dense', 0.01)
        grad_clip = self.config.get('dense', {}).get('grad_clip', 5.0)
        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=lr_dense, momentum=0.9)

        # Last known-good student; restored (with lr reduced 10x) if KD diverges.
        good_state = copy.deepcopy(self.global_model.state_dict())

        for epoch in range(self.epochs):
            synthesizer.synthesize(epoch)
            kd_loss, max_gn, n_skip = kd_train(synthesizer, self.global_model, ensemble_model,
                                               self.criterion, optimizer, self.device, grad_clip=grad_clip)
            finite = all(torch.isfinite(v).all() for v in self.global_model.state_dict().values()
                         if v.is_floating_point())
            if not finite:
                self.global_model.load_state_dict(good_state)
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1
                optimizer.state.clear()
                self.logger.warning(f"DENSE Epoch {epoch+1}: non-finite student weights, restored last good "
                                    f"state and reduced lr to {optimizer.param_groups[0]['lr']:.2e}")
            else:
                good_state = copy.deepcopy(self.global_model.state_dict())
            self.logger.info(f"DENSE Epoch {epoch+1}/{self.epochs} complete. kd_loss={kd_loss:.4f} "
                             f"max_grad_norm={max_gn:.2f} skipped={n_skip}")

        self.logger.info(f"DENSE distillation complete.")

    def __repr__(self) -> str:
        return " DENSE Strategy"
