import torch
from tqdm import tqdm
import json
from utils import KNN

class Trainer:
    def __init__(self, dataset, train_args, model):
        self.model = model
        self.dataset = dataset
        self.train_dataloader = dataset.get_train_dataloader()
        self.test_dataset = dataset.get_test_dataset()

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

    def evaluate(self, n_neighbors):
        W_dist = []
        for idx_X in enumerate(self.test_dataset.dataset):
            print(f'Calculating all distances of sample {idx_X + 1}')
            W_X = []
            for idx_Y, Y in enumerate(self.test_dataset):
                if idx_X == idx_Y:
                    W_X.append(0.)
                else:
                    dist = self.optimal_transport(self.test_dataset.dataset[0][idx_X],
                                                  self.test_dataset.dataset[0][idx_Y])
                    W_X.append(dist.item())
            W_dist.append(W_X)

        with open('distances.json', 'w') as f:
            json.dump(W_dist, f)

        model = KNN(n_neighbors)

        n_class = torch.Tensor(torch.arange(0, 10)).view(10, 1)
        y_train = n_class.expand_as(torch.empty((10, 660))).contiguous().view(6600, 1)
        y_test = n_class.expand_as(torch.empty((10, 220))).contiguous().view(2200, 1)

        count = 0
        f = open('distances.json')
        x_test = json.load(f)
        source_labels = y_test.view(-1, 1)
        count += model.forward(torch.Tensor(x_test), source_labels, y_train)

        return count


    def optimal_transport(self, P, Q):
        return self.model.W_M(P, Q)
