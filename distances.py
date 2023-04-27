import torch
import numpy as np


def mahalanobis_dist(P: torch.Tensor, Q: torch.Tensor, M:torch.Tensor):
    # P: [N, D]
    # Q: [M, D]
    # M: [D, D]
    element_feature_subtract = P.unsqueeze(dim=1) - Q
    return (element_feature_subtract.matmul(M) * element_feature_subtract).sum(dim=-1)
