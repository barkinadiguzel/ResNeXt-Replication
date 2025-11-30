
# Model Hyperparameters
MODEL = {
    "num_classes": 1000,          # Number of output classes (ImageNet)
    "input_channels": 3,          # RGB input
    "cardinality": 32,            # Number of groups in grouped conv
    "bottleneck_width": 4,        # Width of each bottleneck
    "block_repeats": [3, 4, 6, 3] # Number of blocks in conv2-5
}

# Training Hyperparameters
TRAIN = {
    "batch_size": 256,            # Total batch size
    "learning_rate": 0.1,         # Initial learning rate
    "momentum": 0.9,              # SGD momentum
    "weight_decay": 1e-4,         # L2 regularization
    "lr_schedule": [30, 60, 90],  # Epochs to decay learning rate by 10
    "num_epochs": 100,            # Total number of training epochs
    "num_gpus": 8                 # Number of GPUs used
}

# Input Settings
INPUT = {
    "img_size": 224,              # Input image size
    "crop_size": 224,             # Random crop size for training
    "short_side": 256,            # Shorter side resize for center crop
}

# Miscellaneous
SEED = 42                        # Random seed for reproducibility
