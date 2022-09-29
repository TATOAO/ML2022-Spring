import datetime
import gc
import torch.nn as nn
from utiity import *
from config import *
from modal import *
from dataset import LibriDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    # preprocess data
    train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                       concat_nframes=concat_nframes, train_ratio=train_ratio)
    val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                   concat_nframes=concat_nframes, train_ratio=train_ratio)

    # get dataset
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)

    # remove raw feature to save memory
    del train_X, train_y, val_X, val_y
    gc.collect()

    # get dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    # create model, define a loss function, and optimizer
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train()  # set the model to training mode
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        # validation
        if len(val_set) > 0:
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)

                    loss = criterion(outputs, labels)

                    _, val_pred = torch.max(outputs, 1)
                    val_acc += (
                                val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                    val_loss += loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                    val_acc / len(val_set), val_loss / len(val_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
            ))

    # if not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')

    del train_loader, val_loader
    gc.collect()

    # load data
    test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone',
                             concat_nframes=concat_nframes)
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # load model
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    test_acc = 0.0
    test_lengths = 0
    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)

            outputs = model(features)

            _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    with open(save_path, 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))

    end_time = datetime.datetime.now()
    print('运行完毕，程序耗时[%d]秒' % (end_time - start_time).seconds)
