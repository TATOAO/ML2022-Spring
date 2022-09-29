# data prarameters
concat_nframes = 7              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.9               # the ratio of data used for training, the rest will be used for validation
# training parameters
seed = 0                        # random seed
batch_size = 1024                # batch size
num_epoch = 20                   # the number of training epoch
learning_rate = 1e-3          # learning rate
# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 3               # the number of hidden layers
hidden_dim = 256                # the hidden dim
version = 6
weight_decay = 1e-2           # weight decay
model_path = './modelv%d.ckpt'%version     # the path where the checkpoint will be saved
save_path = 'predictionv%d.csv'%version