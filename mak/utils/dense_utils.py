import copy
import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

try:
    from kornia import augmentation
except ImportError:
    augmentation = None

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nc=3):
        self.root = os.path.abspath(root)
        self.images = [os.path.join(self.root, f) for f in os.listdir(self.root) 
                       if f.endswith('.png') and f.startswith('batch_')]
        self.transform = transform
        self.nc = nc

    def __getitem__(self, idx):
        if self.nc == 1:
            img = Image.open(self.images[idx]).convert('L')
        else:
            img = Image.open(self.images[idx]).convert('RGB')
            
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

class LabeledSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nc=3, teacher=None, device='cpu'):
        self.dataset = UnlabeledImageDataset(root, transform=transform, nc=nc)
        self.teacher = teacher
        self.device = device
        self.cache = {}

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if idx in self.cache:
            return img, self.cache[idx]
            
        if self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                # Apply same normalization as during training if needed
                # For simplicity, we assume 'img' is already transformed/normalized
                input_tensor = img.unsqueeze(0).to(self.device)
                logits = self.teacher(input_tensor)
                label = logits.argmax(dim=1).item()
                self.cache[idx] = label
                return img, label
        return img, 0

    def __len__(self):
        return len(self.dataset)

class ImagePool(object):
    def __init__(self, root, nc=3, remove_existing=False):
        self.root = os.path.abspath(root)
        self.nc = nc
        if remove_existing and os.path.exists(self.root):
            import shutil
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)
        batches = [f for f in os.listdir(self.root) if f.startswith('batch_')]
        if batches:
            self._idx = max([int(f.split('_')[1]) for f in batches]) + 1
        else:
            self._idx = 0

    def add(self, imgs):
        if isinstance(imgs, torch.Tensor):
            imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
        
        # Save as a single batch file for simplicity by iteration
        for i, img_np in enumerate(imgs):
            if img_np.shape[0] == 1:
                img = Image.fromarray(img_np[0])
            else:
                # Transpose from CxHxW to HxWxC
                img = Image.fromarray(img_np.transpose(1, 2, 0))
            img.save(os.path.join(self.root, f"batch_{self._idx}_img_{i}.png"))
        self._idx += 1

    def get_dataset(self, transform=None):
        return UnlabeledImageDataset(self.root, transform=transform, nc=self.nc)

class DeepInversionHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.r_feature = 0

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()

class MultiTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

class Ensemble(nn.Module):
    def __init__(self, model_list, p):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.p = p

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += self.p[i] * logits
        return logits_total

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=28, nc=1):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class AdvSynthesizer():
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                 iterations, lr_g, synthesis_batch_size, sample_batch_size,
                 adv, bn, oh, save_dir, dataset, mean=None, std=None):
        super(AdvSynthesizer, self).__init__()
        self.student = student
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.save_dir = save_dir
        self.data_pool = ImagePool(root=self.save_dir, nc=self.img_size[0], remove_existing=False)
        self.teacher = teacher
        self.dataset = dataset.lower()
        self.generator = generator.train()
        
        self.mean = mean if mean is not None else [0.5] * self.img_size[0]
        self.std = std if std is not None else [0.5] * self.img_size[0]
        self.normalize = augmentation.Normalize(self.mean, self.std) if augmentation else None

        if augmentation:
            self.aug = MultiTransform([
                transforms.Compose([
                    augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                    augmentation.RandomHorizontalFlip(),
                ]),
                transforms.Compose([
                    augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                    augmentation.RandomHorizontalFlip(),
                ]),
            ])
        else:
            self.aug = lambda x: [x, x]

        # KD training transforms
        kd_trans_list = []
        if "cifar" in self.dataset:
            kd_trans_list.append(transforms.RandomCrop(32, padding=4))
        kd_trans_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.transform = transforms.Compose(kd_trans_list)

    def get_data(self):
        datasets = self.data_pool.get_dataset(transform=self.transform)
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True)
        return self.data_loader

    def synthesize(self, cur_ep):
        self.teacher.eval()
        # Official code leaves the student in train mode here (no .eval()); this
        # only matters for BN/dropout students.
        self.student.train()

        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=next(self.generator.parameters()).device)
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,), device=z.device).sort()[0]
        
        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g, betas=[0.5, 0.999])
        
        hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(DeepInversionHook(m))

        best_cost = 1e6
        best_inputs = None

        with tqdm(total=self.iterations, desc=f"Synthesize Epoch {cur_ep}", leave=False) as pbar:
            for it in range(self.iterations):
                optimizer.zero_grad()
                inputs = self.generator(z)
                global_view, local_view = self.aug(inputs)
                
                # Apply normalization before passing to models
                if self.normalize:
                    global_view = self.normalize(global_view)
                
                t_out = self.teacher(global_view)
                s_out = self.student(global_view)
                
                loss_bn = sum([h.r_feature for h in hooks]) if hooks else torch.tensor(0.0).to(z.device)
                loss_oh = F.cross_entropy(t_out, targets)
                
                mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()

                loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

                if not torch.isfinite(loss):
                    # Don't push NaN/inf gradients into the generator.
                    pbar.update(1)
                    continue

                if loss.item() < best_cost:
                    best_cost = loss.item()
                    best_inputs = inputs.detach().clone()

                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'bn': f"{loss_bn.item():.4f}"})
                pbar.update(1)

        for h in hooks:
            h.remove()
        
        if best_inputs is not None:
            self.data_pool.add(best_inputs)
            # Save a sample for visual inspection
            vutils.save_image(best_inputs[:100], os.path.join(self.save_dir, f"sample_epoch_{cur_ep}.png"), 
                              normalize=True, nrow=10)

def kd_train(synthesizer, student, teacher, criterion, optimizer, device, grad_clip=None):
    """One KD pass over the synthetic pool.

    With T=20 the T^2-scaled KL yields gradients ~T times larger than CE; on
    BN-free nets (FedNet/AlexNet here) SGD+momentum then diverges to NaN or to
    dead ReLUs (all-zero logits). Gradient-norm clipping and skipping
    non-finite steps prevent that. Returns (mean_loss, max_grad_norm, n_skipped).
    """
    student.train()
    teacher.eval()

    loader = synthesizer.get_data()
    total_loss, n_steps, n_skipped, max_gn = 0.0, 0, 0, 0.0
    for images in loader:
        images = images.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            t_out = teacher(images)
        s_out = student(images)
        loss = criterion(s_out, t_out.detach())
        if not torch.isfinite(loss):
            n_skipped += 1
            continue
        loss.backward()
        max_norm = grad_clip if grad_clip else float('inf')
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)
        if not torch.isfinite(gn):
            optimizer.zero_grad()
            n_skipped += 1
            continue
        max_gn = max(max_gn, gn.item())
        optimizer.step()
        total_loss += loss.item()
        n_steps += 1
    return total_loss / max(n_steps, 1), max_gn, n_skipped
