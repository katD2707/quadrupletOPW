from qopw import QOP
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, train_args, model_args):
        self.qopw = QOP(
            dim=model_args.dim,
            std=model_args.std,
            lambda_2=model_args.lambda_2,
            lambda_3=model_args.lambda_3,
            verbose=model_args.verbose,
            p_norm=model_args.p_norm,
            sinkhorn_maxIter=model_args.sinkhorn_maxIter,
            tol=model_args.tol,
            alpha=model_args.alpha,
            beta=model_args.beta,
            )

        self.n_epochs = train_args.n_epochs
        self.lr = train_args.learning_rate
        self.eps = train_args.eps
        self.loss_after_epoch = train_args.loss_after_epoch

        self.P_train = torch.randn(30, 20, 128)
        self.Q_train = torch.randn(30, 40, 128)
        self.R_train = torch.randn(30, 60, 128)
        self.S_train = torch.randn(30, 80, 128)

        self.losses = []

    def training_one_epoch(self):
        loss = torch.Tensor([0.])
        for i in range(self.P_train.size(0)):
            loss += self.qopw(self.P_train[i],
                              self.Q_train[i],
                              self.R_train[i],
                              self.S_train[i],
                              )
        loss.backward()

        with torch.no_grad():
            self.qopw.M -= self.lr * self.qopw.M.grad

        self.losses.append(loss.cpu().detach())

    def train(self):
        for epoch in tqdm(range(self.n_epochs)):
            if self.qopw.M.grad is not None:
                self.qopw.M.grad.zero_()

            self.training_one_epoch()

            if epoch % self.loss_after_epoch == 0:
                print(f'Loss after {epoch} epochs: {self.losses[-1]}')

    def optimal_transport(self, P, Q):
        return self.qopw.W_M(P, Q)


