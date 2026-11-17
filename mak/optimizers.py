from torch.optim import Optimizer

class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cv, client_cv):
        # Move server_cv and client_cv to the same device as model parameters
        device = self.param_groups[0]["params"][0].device
        server_cv = [cv.to(device) for cv in server_cv]
        client_cv = [cv.to(device) for cv in client_cv]
        
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cv, client_cv):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])
