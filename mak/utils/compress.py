import torch
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector, PMatBlockDiag, PMatEKFAC
import numpy as np
import copy
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# Quantizing each layer of a neural network
def quantize_layer(v, model, s):
    v_copy = v.clone().detach()
    count_prev = 0
    for parameter in model.parameters():
        x = torch.numel(parameter)
        v_copy[count_prev:count_prev +
               x] = quantize(v[count_prev:count_prev+x], s)
        count_prev += x

    return v_copy


# For quantizing tensor up to 2**(s+1) bits
def quantize(A, s):

    levels = 2**(s)-1
    d1 = 0
    m = torch.max(torch.abs(A))
    if m == 0:
        # A is all-zero: nothing to quantize, and dividing by m would be 0/0 = NaN.
        # This is common for a client's Fisher-diagonal slice on parameters its
        # local data never touches, especially under extreme non-IID splits with
        # many clients — silently propagating NaN here corrupts the whole merge.
        return A.clone()
    C = A/m
    C_1 = torch.round(levels*C)
    C_2 = m*C_1/levels
    return C_2


# Per-layer Tikhonov-style floor on a flattened Fisher-diagonal vector.
#
# Per-client Fisher estimates get noisy — and can be exactly zero for whole
# layers — when a client's local data barely touches those parameters
# (extreme non-IID splits, many clients, or both, all shrink the effective
# per-client-per-class sample count). Two things go wrong downstream without
# a floor: quantize()'s divide-by-max on an all-zero layer is 0/0 = NaN, and
# even short of that, a merge formula like fisher_avg = grad_sum / F_sum
# amplifies noise sharply whenever F_sum is small-but-nonzero. Flooring each
# client's Fisher diagonal per-layer (mirroring quantize_layer's own
# per-parameter-tensor loop, since natural Fisher scale varies a lot across
# layers) fixes both: it's a ridge/EWC-style regularizer, not just a NaN
# patch.
def regularize_fisher(F, model, eps_rel=1e-3, eps_abs=1e-8):
    F_reg = F.clone()
    count_prev = 0
    for parameter in model.parameters():
        x = torch.numel(parameter)
        layer_slice = F[count_prev:count_prev + x]
        scale = layer_slice.abs().max().item()
        floor = max(eps_rel * scale, eps_abs)
        F_reg[count_prev:count_prev + x] = layer_slice + floor
        count_prev += x
    return F_reg


# Compressing a matrix using both quantization and low-rank decomposition


def compress_mat(A, s_quant, s_svd):

    A_svd = torch.linalg.svd(A)
    b = A.shape[0]
    U = quantize(A_svd[0][:, 0:s_svd], s_quant)
    S = quantize(A_svd[1][0:s_svd], s_quant)
    V = quantize(A_svd[2][0:s_svd], s_quant)
    A_low_rank = torch.mm(torch.mm(U, torch.diag(S)), V)

    return A_low_rank


def compress_regmean_gram(gram_matrix_list, model):

    n = len(gram_matrix_list)
    d = parameters_to_vector(model.parameters()).numel()
    d1 = 0
    gram_matrix = gram_matrix_list[0]

    # Computing the total number of parameters in the gram matrix of each layer
    for layer_id in gram_matrix.keys():
        d1 = d1 + torch.numel(gram_matrix[layer_id])

    print("Ratio of parameters in RegMean to Vanilla Avg.", (d1+d)/d)

    '''
    Specify factor by which you want to compress the Gram matrix using quantization (quant_comp) and then 
    compute the compression factor for svd. Note that after compression we want the number of bits to be close 
    to 32*d/2 bits. The remaining 32*d/2 bits are used for sending the model weights
    '''
    quant_comp = 4
    svd_comp = (d1/quant_comp)/((d/2))

    s_quant = int(32/quant_comp)

    r1 = 0
    for layer_id in gram_matrix.keys():
        b1 = gram_matrix[layer_id].shape[0]
        r1 = r1 + 2*b1 + 1
    s_svd = int((d1)/(svd_comp*r1))

    # Compute number of parameters after compression
    d_new = 0
    for layer_id in gram_matrix.keys():
        b1 = gram_matrix[layer_id].shape[0]
        d_new = d_new + 2*b1*min(b1, s_svd)*(s_quant+1) + \
            min(b1, s_svd)*(s_quant+1) + 3*32

    '''
    2*b1*min(b1,s_svd)*(s_quant+1) is the number of bits for the left and right singular vectors. 
    min(b1,s_svd)*(s_quant+1) is the number of bits for the singular values. 3*32 bits are added to communicate
    the max element for quantization.
    '''

    # We aim to get this ratio close to 1
    print("Ratio after compression", (d_new+16*d)/(32*d))

    for i in range(n):
        for layer_id in gram_matrix_list[i].keys():
            gram_matrix_list[i][layer_id] = compress_mat(
                gram_matrix_list[i][layer_id], s_quant, s_svd)

    # d_new is the compressed payload size (in bits) of ONE client's Gram
    # matrices -- compress_mat reconstructs a full-shape tensor for
    # numerical simulation, so it doesn't actually shrink in memory here,
    # but this is the real bit-budget a client would transmit. Every
    # client shares the same model architecture (same layer shapes), so
    # the total across all n clients is n * d_new bits.
    return d_new * n


def compress_fisher_kfac(args, F_kfac_list_org, models, model_vectors, dataset_train, p, num_classes, client_indices, device):

    F_kfac_list = []
    n = len(F_kfac_list_org)
    d = parameters_to_vector(models[0].parameters()).numel()
    d1 = 0

    '''
    Creating dummy KFAC for each client. Later the values of the dummy KFAC will be
    assigned to the compressed KFAC
    '''
    for i in range(n):
        # Pick one random index from client_indices[i]
        choice_idx = np.random.choice(client_indices[i], 1)
        dataset_train_subset = torch.utils.data.Subset(
            dataset_train, choice_idx)
        data_loader = DataLoader(dataset_train_subset,
                                 batch_size=args['client']['test_batch_size'], shuffle=True)
        F_kfac = FIM(model=models[i], loader=data_loader, representation=PMatKFAC, device=device,
                     variant='classif_logits', n_output=num_classes)

        # Computing the total number of parameters in the KFAC matrices of each layer
        if (i == 0):
            for layer_id in F_kfac.data.keys():
                d1 = d1 + \
                    torch.numel(F_kfac.data[layer_id][0]) + \
                    torch.numel(F_kfac.data[layer_id][1])

        F_kfac_list.append(F_kfac)

    # Compression starts here
    for i in range(n):

        print(i)
        F_kfac = F_kfac_list[i]
        F_kfac_org = F_kfac_list_org[i]

        r1 = 0
        if (i == 0):
            print("Ratio of parameters in Fisher KFAC to Vanilla Avg.", (d1+d)/d)

        '''
        Specify factor by which you want to compress the KFAC matrices using quantization (quant_comp) and then 
        compute the compression factor for svd. Note that after compression we want the number of bits to be close 
        to 32*d/2 bits. The remaining 32*d/2 bits are used for sending the model weights
        '''
        quant_comp = 4
        svd_comp = (d1/quant_comp)/((d/2))
        s_quant = int(32/quant_comp)

        if (i == 0):
            for layer_id in F_kfac.data.keys():
                b1 = F_kfac.data[layer_id][0].shape[0]
                b2 = F_kfac.data[layer_id][1].shape[0]
                r1 = r1 + 2*b1 + 2*b2 + 2

            s_svd = int((d1)/(svd_comp*r1))
            d_new = 0
            for layer_id in F_kfac.data.keys():
                b1 = F_kfac.data[layer_id][0].shape[0]
                b2 = F_kfac.data[layer_id][1].shape[0]
                d_new = d_new + 2*b1*min(b1, s_svd)*(s_quant+1) + 2*b2*min(b2, s_svd)*(s_quant+1) \
                    + min(b1, s_svd)*(s_quant+1) + \
                    min(b2, s_svd)*(s_quant+1) + 6*32

            '''
            2*b1*min(b1,s_svd)*(s_quant+1) is the number of bits for the left and right singular vectors of the
            left Kroncker matrix. 2*b2*min(b2,s_svd)*(s_quant+1) is the number of bits for the left and right 
            singular vectors of the right Kroncker matrix. min(b1,s_svd)*(s_quant+1) is the number of bits 
            for the singular values of the left Kronecker matrix. min(b2,s_svd)*(s_quant+1) is the number of bits 
            for the singular values of the right Kronecker matrix. 6*32 bits is added to communicate
            the max element for quantization.
            '''

            # We aim to get this ratio close to 1
            print("Ratio after compression", (d_new+16*d)/(32*d))

        for layer_id in F_kfac.data.keys():

            F_kfac.data[layer_id] = list(F_kfac.data[layer_id])
            F_kfac.data[layer_id][0] = compress_mat(
                F_kfac_org.data[layer_id][0], s_quant, s_svd)
            F_kfac.data[layer_id][1] = compress_mat(
                F_kfac_org.data[layer_id][1], s_quant, s_svd)

        # Quantizing the model
        model_vector_quantized = quantize_layer(
            model_vectors[i], models[0], 15)
        model_proxy = copy.deepcopy(models[0])
        vector_to_parameters(model_vector_quantized, model_proxy.parameters())

        v_base_model = PVector.from_model(model_proxy)
        v = F_kfac.mv(v_base_model)
        v = v.__rmul__(1/p[i])

        if (i == 0):
            grad_avg = v
        else:
            grad_avg = grad_avg.__add__(v)

    return F_kfac_list, grad_avg


# def get_dot_product(F_list, w, p):
#     n = len(F_list)
#     for i in range(n):
#         v_i = F_list[i].mv(w)
#         v_i = v_i.__rmul__(1.0/p[i])
#         if i == 0:
#             v = v_i
#         else:
#             v = v.__add__(v_i)
#     return v

def get_dot_product(F_mat_list, w, p):
    n = len(F_mat_list)
    v = F_mat_list[0].mv(w)
    v = v.__rmul__(1/p[0])
    for i in range(1, n):
        x = F_mat_list[i].mv(w)
        x = x.__rmul__(1/p[i])
        v = v.__add__(x)
    return v


def element_mul(self, other):
    if self.dict_repr is not None and other.dict_repr is not None:
        v_dict = dict()
        for l_id, l in self.layer_collection.layers.items():
            if l.bias is not None:
                v_dict[l_id] = (self.dict_repr[l_id][0] *
                                other.dict_repr[l_id][0],
                                self.dict_repr[l_id][1] *
                                other.dict_repr[l_id][1])
            else:
                v_dict[l_id] = (self.dict_repr[l_id][0] *
                                other.dict_repr[l_id][0],)
        return PVector(self.layer_collection, dict_repr=v_dict)
    elif self.vector_repr is not None and other.vector_repr is not None:
        return PVector(self.layer_collection,
                       vector_repr=self.vector_repr+other.vector_repr)
    else:
        return PVector(self.layer_collection,
                       vector_repr=(self.get_flat_representation() +
                                    other.get_flat_representation()))


def inv_sqrt(self, epsilon):

    if self.dict_repr is not None:
        v_dict = dict()
        for l_id, l in self.layer_collection.layers.items():
            if l.bias:
                v_dict[l_id] = (1/torch.sqrt(self.dict_repr[l_id][0] + epsilon),
                                1/torch.sqrt(self.dict_repr[l_id][1] + epsilon))
            else:
                v_dict[l_id] = (
                    1/torch.sqrt(self.dict_repr[l_id][0] + epsilon),)
        return PVector(self.layer_collection, dict_repr=v_dict)
    else:
        return PVector(self.layer_collection,
                       vector_repr=x * self.vector_repr)
