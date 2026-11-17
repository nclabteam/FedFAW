import torch
from nngeometry.object import PMatKFAC, PMatDiag, PVector

try:
    from laplace.baselaplace import ParametricLaplace
except ImportError:
    ParametricLaplace = None


def _laplace_posterior_bytes(la):
    """
    Size of a fitted laplace-torch posterior (la.H). For hessian_structure=
    'kron', H is a KronDecomposed (eigenvectors + eigenvalues + deltas per
    layer, post-.fit()); for 'diag'/'full', H is a plain tensor. This is
    exactly what a client must transmit to the server for FedBEns-style
    aggregation (MixtureModel_kron_nb.log_prob uses it directly) -- not just
    the model weights, which is all get_size_mb counted before this.
    """
    size_bytes = 0
    H = getattr(la, 'H', None)
    if H is None:
        return 0
    if isinstance(H, torch.Tensor):
        return H.numel() * H.element_size()
    for group in getattr(H, 'eigenvectors', []):
        tensors = group if isinstance(group, (list, tuple)) else [group]
        for t in tensors:
            size_bytes += t.numel() * t.element_size()
    for group in getattr(H, 'eigenvalues', []):
        tensors = group if isinstance(group, (list, tuple)) else [group]
        for t in tensors:
            size_bytes += t.numel() * t.element_size()
    deltas = getattr(H, 'deltas', None)
    if deltas is not None:
        size_bytes += deltas.numel() * deltas.element_size()
    return size_bytes


def get_size_mb(obj):
    """Calculates size of model state_dict, list of tensors, or Fisher matrices in MB."""
    size_bytes = 0

    if isinstance(obj, torch.Tensor):
        size_bytes = obj.numel() * obj.element_size()
    elif isinstance(obj, dict):
        for val in obj.values():
            size_bytes += get_size_bytes(val)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for val in obj:
            size_bytes += get_size_bytes(val)
    elif isinstance(obj, PMatKFAC):
        # PMatKFAC.data is a dict of {layer_id: (factor1, factor2)}
        for factors in obj.data.values():
            for factor in factors:
                size_bytes += factor.numel() * factor.element_size()
    elif isinstance(obj, PMatDiag):
        # If it's the object itself, not just the diag vector
        diag = obj.get_diag()
        size_bytes = diag.numel() * diag.element_size()
    elif isinstance(obj, PVector):
        vec = obj.get_flat_representation()
        size_bytes = vec.numel() * vec.element_size()
    elif ParametricLaplace is not None and isinstance(obj, ParametricLaplace):
        size_bytes = _laplace_posterior_bytes(obj)

    return size_bytes / (1024 * 1024)

def get_size_bytes(obj):
    """Helper to recursively calculate size in bytes."""
    size_bytes = 0
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    elif isinstance(obj, dict):
        for val in obj.values():
            size_bytes += get_size_bytes(val)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for val in obj:
            size_bytes += get_size_bytes(val)
    elif isinstance(obj, PMatKFAC):
        for factors in obj.data.values():
            for factor in factors:
                size_bytes += factor.numel() * factor.element_size()
    elif isinstance(obj, PMatDiag):
        diag = obj.get_diag()
        size_bytes = diag.numel() * diag.element_size()
    elif isinstance(obj, PVector):
        vec = obj.get_flat_representation()
        size_bytes = vec.numel() * vec.element_size()
    elif ParametricLaplace is not None and isinstance(obj, ParametricLaplace):
        size_bytes = _laplace_posterior_bytes(obj)
    elif hasattr(obj, 'parameters'): # It's a model
        for param in obj.parameters():
            size_bytes += param.numel() * param.element_size()
    return size_bytes
