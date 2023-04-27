import torch
from distances import mahalanobis_dist
import numpy as np
from torch import nn


class QOP(nn.Module):
    def __init__(self, dim, std, lambda_2, lambda_3, verbose, p_norm="inf",
               sinkhorn_maxIter=20, tol=1e-5, alpha=None, beta=None, *args, **kwargs):
        super(QOP, self).__init__()
        self.std = std
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.p_norm = p_norm
        self.sinkhorn_maxIter = sinkhorn_maxIter
        self.tol = tol

        self.M = nn.Parameter(torch.randn(dim, dim), requires_grad=True)

    def W_M(self, P, Q):
        m = P.size(0)
        n = Q.size(0)

        dist = mahalanobis_dist(P, Q, self.M)

        alpha = self.alpha
        beta = self.beta
        if alpha == "None":
            alpha = torch.arange(1, m + 1).view(m, 1) / m
        if beta == "None":
            beta = torch.arange(1, n + 1).view(n, 1) / n

        relative_pos = alpha.unsqueeze(1) - beta
        relative_pos = relative_pos.squeeze()

        prior = 1 / (self.std * (2 * np.pi) ** (1 / 2)) * torch.exp(-(relative_pos.abs() / (1 / n ** 2 + 1 / m ** 2) ** (1 / 2)) ** 2 / (2 * (self.std ** 2)))

        with torch.no_grad():
            K = prior * torch.exp(-1 / self.lambda_3 * (dist - self.lambda_2 / (relative_pos ** 2 + 1)))

            u, v = self.sinkhorn_iterative(K, alpha, beta)

            F = (u * K) * v.T

        return (F * dist).sum()

    def sinkhorn_iterative(self, K, alpha: torch.Tensor, beta: torch.Tensor):
        ainvK = K / alpha  # [N, M]

        iter = 0
        u = alpha.clone()

        while iter < self.sinkhorn_maxIter:
            u = 1. / torch.matmul(ainvK, (beta / (torch.matmul(K.T, u))))
            iter += 1
            if iter % 20 == 1 or iter == self.sinkhorn_maxIter:
                v = beta / torch.matmul(K.T, u)  # [M, 1]
                u = 1 / torch.matmul(ainvK, v)  # [N, 1]

                criterion = torch.sum(torch.abs(v * torch.matmul(K.T, u) - beta), dim=0)
                criterion = criterion.norm(p=float(self.p_norm))
                if criterion.abs().item() < self.tol:
                    break

                iter += 1
                if self.verbose > 0:
                    print(f"Iteration : {iter}, Criterion: {criterion}")

        return u, v

    def forward(self, P, Q, R, S):
        loss = self.W_M(P, Q) - self.W_M(R, S)
        return loss


