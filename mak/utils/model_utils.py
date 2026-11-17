import os
import numpy as np
import torch
from mak.models import *


def get_network(model, channel, num_classes, im_size=(32, 32), device='cpu', seed=42, use_pretrained = False, pretrained_path = None, input_dim=None):
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes, input_dim=input_dim)
    elif model == 'HARNet':
        net = HARNet(num_classes=num_classes, input_dim=input_dim)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        if(use_pretrained == True):
            print ("Using pretrained model")
            pre_net = ResNet18_tiny(200)
            pre_net.load_state_dict(torch.load(pretrained_path))
            net = ResNet18_mod(num_classes)
            pre_net_dict = pre_net.state_dict()
            net_dict = net.state_dict()
            for layer in pre_net_dict:
                net_dict[layer] = pre_net_dict[layer]
            net.load_state_dict(net_dict)
        else:
            net = ResNet18(num_classes)

    elif model == 'FedNet':
        net = FedNet(channel=channel, num_classes=num_classes, input_shape=im_size)
    elif model == 'FedAvgCNN':
        net = FedAvgCNN(channel=channel, num_classes=num_classes, input_shape=im_size)
    else:
        raise ValueError('Unknown model: %s' % model)

    net = net.to(device)

    return net


def get_default_convnet_setting():
    """
    Provides default settings for the ConvNet architecture.

    Returns:
        tuple: (net_width, net_depth, net_act, net_norm, net_pooling)
    """
    net_width = 128
    net_depth = 3
    net_act = 'relu'
    net_norm = 'batchnorm'
    net_pooling = 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def save_item(item, role, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, role + "_" + item_name + ".pt"))


def load_item(role, item_name, item_path=None):
    try:
        return torch.load(os.path.join(item_path, role + "_" + item_name + ".pt"), weights_only=False)
    except FileNotFoundError:
        print(role, item_name, 'Not Found')
        return None


def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = F.normalize(proto / len(proto_list), dim=0)
        else:
            protos[label] = F.normalize(proto_list[0], dim=0)

    return protos

# https://github.com/NeuralCollapseApplications/ImbalancedLearning/blob/main/models/resnet.py#L347


def generate_random_orthogonal_matrix(feat_in, num_classes):
    a = np.random.random(size=(feat_in, num_classes))
    P, _ = np.linalg.qr(a)
    P = torch.tensor(P).float()
    assert torch.allclose(torch.matmul(P.T, P), torch.eye(
        num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
    return P


def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)

# from FedKTL repo
# split an original model into a base and a head


class BaseHeadSplit(nn.Module):
    def __init__(self, model, feature_dim, num_classes, args, cid):
        super().__init__()

        # self.base = eval(model)
        self.base = model
        head = None  # you may need more code for pre-existing heterogeneous heads
        if hasattr(self.base, 'heads'):
            head = self.base.heads
            self.base.heads = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'head'):
            head = self.base.head
            self.base.head = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'fc'):
            head = self.base.fc
            self.base.fc = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'classifier'):
            head = self.base.classifier
            self.base.classifier = nn.AdaptiveAvgPool1d(feature_dim)
        else:
            raise ('The base model does not have a classification head.')

        if hasattr(args, 'heads'):
            self.head = eval(model)
        # elif 'vit' in args.models[cid % len(args.models)]:
        #     self.head = nn.Sequential(
        #         nn.Linear(args.feature_dim, 768),
        #         nn.Tanh(),
        #         nn.Linear(768, args.num_classes)
        #     )
        else:
            self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out
