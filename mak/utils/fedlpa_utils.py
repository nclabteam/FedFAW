import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.optim.optimizer import Optimizer

class KFACOptimizer(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9, stat_decay=0.95, damping=1e-3, 
                 kl_clip=1e-2, weight_decay=3e-3, TCov=1, TScal=10, TInv=100):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, momentum=momentum, stat_decay=stat_decay, damping=damping,
                        kl_clip=kl_clip, weight_decay=weight_decay, TCov=TCov, TScal=TScal, TInv=TInv)
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.model = model
        self.modules = []
        self.module2name = {}
        self._prepare_model()
        self.steps = 0
        self.m_aa = {}
        self.m_gg = {}
        self.hooks = []
        self._register_hooks()

    def _prepare_model(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                self.modules.append(module)
                self.module2name[module] = name

    def _register_hooks(self):
        for module in self.modules:
            handle_f = module.register_forward_pre_hook(self._save_input)
            handle_b = module.register_backward_hook(self._save_grad_output)
            self.hooks.append(handle_f)
            self.hooks.append(handle_b)

    def _save_input(self, module, input):
        if not hasattr(self, 'acc_stats') or not self.acc_stats:
            return
        aa = input[0].data
        if isinstance(module, nn.Linear):
            # aa: [batch_size, in_features]
            if module.bias is not None:
                aa = torch.cat([aa, torch.ones(aa.size(0), 1).to(aa.device)], dim=1)
        elif isinstance(module, nn.Conv2d):
            # aa: [batch_size, in_channels, h, w]
            aa = F.unfold(aa, module.kernel_size, padding=module.padding, stride=module.stride)
            # aa: [batch_size, in_channels * k * k, h_out * w_out]
            aa = aa.transpose(1, 2).contiguous().view(-1, aa.size(1))
            if module.bias is not None:
                aa = torch.cat([aa, torch.ones(aa.size(0), 1).to(aa.device)], dim=1)

        
        # covariance A = E[aa^T]
        A = aa.t() @ aa / aa.size(0)
        self.m_aa[module] = A

    def _save_grad_output(self, module, grad_input, grad_output):
        if not hasattr(self, 'acc_stats') or not self.acc_stats:
            return
        gg = grad_output[0].data
        if isinstance(module, nn.Linear):
            # gg: [batch_size, out_features]
            pass
        elif isinstance(module, nn.Conv2d):
            # gg: [batch_size, out_channels, h_out, w_out]
            gg = gg.transpose(1, 2).transpose(2, 3).contiguous().view(-1, gg.size(1))
        
        # covariance G = E[gg^T]
        # In Pytorch, gradients are typically averaged over the batch (and spatial dims for Conv2d).
        # To get the Empirical Fisher (1/N * sum grad*grad^T), we need to multiply by N.
        G = (gg.t() @ gg) * gg.size(0)
        self.m_gg[module] = G


    def step(self, closure=None):
        pass

def newton_solve(fed_avg_freqs, Ak, Bk, Z, device, iterations=1000):
    # Newton approximation for solving Sum(freq_k * Bk @ HatM @ Ak) = Z
    with torch.no_grad():
        avg_A = sum([fed_avg_freqs[idx] * Ak[idx] for idx in range(len(Ak))])
        avg_B = sum([fed_avg_freqs[idx] * Bk[idx] for idx in range(len(Bk))])
        
        eps = 1e-10
        InverseSumA = torch.inverse(avg_A + eps * torch.eye(avg_A.size(0)).to(device))
        InverseSumB = torch.inverse(avg_B + eps * torch.eye(avg_B.size(0)).to(device))
        
        HatM = torch.zeros_like(Z).to(device)
    
        # Iterative Newton steps for better stability
        for i in range(iterations):
            # f(M) = Sum(p_k * B_k @ M @ A_k) - Z
            MuZ = sum([fed_avg_freqs[idx] * Bk[idx] @ HatM @ Ak[idx] for idx in range(len(Ak))])
            fHatM = MuZ - Z
            error = fHatM.abs().mean()
            if error < 1e-10:
                break
            # Newton refinement step
            HatM = HatM - InverseSumB @ fHatM @ InverseSumA
            
    return HatM.detach()


def posterior_loss(f_global, f_local, mu_global, net, device):
    """
    Alignment term as defined in FedLPA. 
    L_align = 0.5 * Tr((M - M_global)^T (F_global - F_local) (M - M_global))
    """
    result = 0.0
    # Use named_parameters to ensure differentiability
    params_dict = dict(net.named_parameters())
    
    for params_name in f_local:
        if params_name in f_global.keys() and len(f_global[params_name]) > 0:
            weight_name = params_name + '.weight'
            bias_name = params_name + '.bias'
            
            if weight_name not in params_dict:
                continue
                
            weight_local = params_dict[weight_name]
            weight_global = mu_global[weight_name].to(device)
            
            # Reshape Conv weights: [out, in, k, k] -> [out, in*k*k]
            if len(weight_local.shape) == 4:
                out_c = weight_local.shape[0]
                M_local_W = weight_local.reshape(out_c, -1)
                M_global_W = weight_global.reshape(out_c, -1)
            else:
                M_local_W = weight_local
                M_global_W = weight_global
                
            if bias_name in params_dict:
                bias_local = params_dict[bias_name]
                bias_global = mu_global[bias_name].to(device)
                M_local = torch.cat([M_local_W, bias_local.unsqueeze(1)], dim=1)
                M_global = torch.cat([M_global_W, bias_global.unsqueeze(1)], dim=1)
            else:
                M_local = M_local_W
                M_global = M_global_W
            
            diff = M_local - M_global
            
            # Precision difference
            term_G = f_global[params_name][1].to(device) - f_local[params_name][1].to(device)
            term_A = f_global[params_name][0].to(device) - f_local[params_name][0].to(device)
            
            # Quadratic form Tr(diff.T @ term_G @ diff @ term_A)
            # Equivalent bit-wise: (diff * (term_G @ diff @ term_A)).sum()
            val = (diff * (term_G @ diff @ term_A)).sum()
            
            # Repo uses 'if ploss > 0' check to handle non-PSD precision differences
            if val > 0:
                result += val

            
    return 0.5 * result
