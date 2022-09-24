config = {
    'seed': 220418,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-5,
    'weight_decay': 1e-4,
    'early_stop': 500,    # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model-v10.ckpt',  # Your model will be saved here.
    'pred_path': 'pred-v11.csv'  # Your model will be saved here.
}