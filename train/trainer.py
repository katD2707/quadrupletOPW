import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, dataset, train_args, model):
        self.model = model
        self.dataset = dataset
        self.train_dataloader = dataset.get_train_dataloader()
        self.test_dataloader = dataset.get_test_dataloader()

        self.n_steps = train_args.n_steps
        self.lr = train_args.learning_rate
        self.eps = train_args.eps
        self.eval_strategy = train_args.eval_strategy
        self.eval_steps = train_args.eval_steps
        self.update_per_steps = train_args.update_per_steps

    def train(self):
        loss = torch.Tensor([0.]).cuda()
        prev_M = 0
        train_data = iter(self.train_dataloader)
        for step in tqdm(range(self.n_steps)):
            data = next(train_data)
            if self.model.M.grad is not None:
                self.model.M.grad.zero_()
            loss += self.model(data[0][0].cuda(),
                               data[1][0].cuda(),
                               data[0][0].cuda(),
                               data[2][0].cuda(),
                               )
            loss_current = loss.detach() / (step + 1)
            if (step + 1) % self.update_per_steps == 0:
                loss = loss / (step + 1)
                loss.backward()

                with torch.no_grad():
                    self.model.M -= self.lr * self.model.M.grad
                    self.model.M.clamp_(min=0)
                    if abs(self.model.M - prev_M).abs().sum() < 1e-16:
                        break
                    else:
                        prev_M = self.model.M.detach()
                loss = torch.Tensor([0.]).cuda()

            if (step + 1) % self.eval_steps == 0:
                print(f'Loss after {step + 1} steps: {loss_current}')

    def evaluate(self):
        pass

    def optimal_transport(self, P, Q):
        return self.model.W_M(P, Q)
