import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, dataset, train_args, model):
        self.model = model
        self.dataset = dataset
        self.train_dataloader = dataset.get_train_dataloader()
        self.test_dataloader = dataset.get_test_dataloader()

        self.n_epochs = train_args.n_epochs
        self.lr = train_args.learning_rate
        self.eps = train_args.eps
        self.eval_strategy = train_args.eval_strategy
        self.eval_steps = train_args.eval_steps

        self.losses = []

    def training_one_epoch(self):
        loss = torch.Tensor([0.])
        N = 0
        for idx, data in enumerate(self.train_dataloader):
            N += 1
            loss += self.model(data[0].cuda(),
                               data[1].cuda(),
                               data[0].cuda(),
                               data[2].cuda(),
                               )
        loss = loss / N
        loss.backward()

        with torch.no_grad():
            self.model.M -= self.lr * self.model.M.grad
            self.model.M = torch.clip(self.model.M, min=0)
        self.losses.append(loss.cpu().detach())

    def train(self):
        for epoch in tqdm(range(self.n_epochs)):
            if self.model.M.grad is not None:
                self.model.M.grad.zero_()

            self.training_one_epoch()

            if epoch % self.eval_steps == 0:
                print(f'Loss after {epoch} epochs: {self.losses[-1]}')

    def optimal_transport(self, P, Q):
        return self.model.W_M(P, Q)
