import torch
import torch.nn as nn
from ..utils.optim import get_lr_scheduler, get_optimizer
from .model import Model

#from opacus import PrivacyEngine

class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes)

    def forward(self, x):
        return self.fc(x)

    def reset_parameters(self):
        self.fc.reset_parameters()
        


class LinearRegressor(Model):
    def __init__(self, criterion, metric, device, input_dimension, num_classes, iterator, args,
                 optimizer_name="sgd", lr_scheduler="constant", initial_lr=1e-3, epoch_size=1, coeff=1):
        super(LinearRegressor, self).__init__()

        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.coeff = coeff

        self.net = LinearLayer(input_dimension, num_classes).to(self.device)
        for param in self.net.parameters():
            print(param.data.shape)
            print(input_dimension)

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

        if args.DP:
            self.sample_rate = min(args.bz / len(iterator.dataset), 1)

            #TODO deal  with the fit epoch case
            self.privacy_engine = PrivacyEngine(
                self.net,
                sample_rate=self.sample_rate,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                target_epsilon=args.epsilon,
                epochs=int(self.sample_rate*args.num_local_steps*args.num_rounds),
                max_grad_norm=args.max_grad_norm,
            )

            self.privacy_engine.attach(self.optimizer)
            print(len(iterator.dataset), self.privacy_engine.noise_multiplier)



    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for x, y in iterator:
            self.optimizer.zero_grad()

            y = y.view(-1)

            predictions = self.net(x).view(-1)

            loss = self.coeff * self.criterion(predictions, y)

            mape = self.metric(predictions, y)

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += mape.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
    
    def fit_batch(self, iterator, update=True):
        self.net.train()

        x, y = next(iter(iterator))

        self.optimizer.zero_grad()

        y = torch.tensor(y, device=self.device)
        y = y.double().view(-1)

        predictions = self.net(x).double().view(-1)

        loss = self.coeff * self.criterion(predictions, y)

        acc = self.metric(predictions, y)

        loss.backward()

        if update:
            self.optimizer.step()
            self.lr_scheduler.step()
        """
        grads = 0
        for param in self.net.parameters():
            grads += torch.linalg.norm(param.grad)**2
        grad_norm = torch.sqrt(grads)
        if grad_norm >= 10:
            print(grad_norm)
        """
        batch_loss = loss.item()
        batch_acc = acc.item()

        return batch_loss, batch_acc

    def evaluate_iterator(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.eval()
        total = 0
        with torch.no_grad():
            for x, y in iterator:
                predictions = self.net(x).view(-1)

                y = y.view(-1)

                loss = self.criterion(predictions, y)

                acc = self.metric(predictions, y)

                epoch_loss += loss.item()*len(y)
                epoch_acc += acc.item()*len(y)
                total += len(y)

        return epoch_loss / total, epoch_acc / total
