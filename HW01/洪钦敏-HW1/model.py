import math
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# tensorboard --logdir=D:\projects\ML2022-Spring\HW01\洪钦敏-HW1\runs
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        d1 = 64
        d2 = 16
        d3 = 8
        layer = 2
        # TODO: modify model's structure, be aware of dimensions.
        if 3 == layer:
            self.layers = nn.Sequential(
                # nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, d1),
                nn.ReLU(),
                # nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
                nn.Linear(d1, d2),
                # nn.ReLU(),
                nn.LeakyReLU(),
                # nn.Dropout(0.2),
                nn.Linear(d2, 1)
            )
        elif 2 == layer:
            self.layers = nn.Sequential(
                # 重要的改动 https://blog.csdn.net/qq_23262411/article/details/100175943
                # 然而没啥用貌似
                # nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, d1),
                # nn.BatchNorm1d(d1),
                nn.LeakyReLU(),
                # nn.Dropout(0.2),
                nn.Linear(d1, 1)
            )
        elif 1 == layer:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 1)
            )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    # optimizer = torch.optim.ASGD(model.parameters(), lr=config['learning_rate'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=config['weight_decay'])
    optimizer = torch.optim.Adam(model.parameters(), betas=[0.9, 0.999], lr=config['learning_rate'], eps=1e-6,
                                 weight_decay=config['weight_decay'])

    writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            # 验证的时候无需计算梯度下降
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        # 训练一定次数后没有更好的loss 则退出训练
        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
