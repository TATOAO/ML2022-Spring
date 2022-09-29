# data prarameters
concat_nframes = 3              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.9               # the ratio of data used for training, the rest will be used for validation

version = 2
# training parameters
seed = 210418                        # random seed
batch_size = 512                # batch size
num_epoch = 20                   # the number of training epoch
learning_rate = 0.0001          # learning rate
model_path = './modelv%d.ckpt'%version     # the path where the checkpoint will be saved
save_path = 'predictionv%d.csv'%version

# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 1               # the number of hidden layers
hidden_dim = 256                # the hidden dim