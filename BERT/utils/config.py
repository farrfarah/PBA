import torch
from argparse import Namespace

# Define configuration for the project
config = Namespace(
    # Paths
    bert_path="bert-base-uncased",
    train_data_path="ag_news_csv/train.csv",
    test_data_path="ag_news_csv/test.csv",
    checkpoint_dir="checkpoints/model.pt",

    # Model parameters
    enc_method="rnn",  # Encoder method
    bert_dim=128,
    hidden_size=8,
    out_size=64,
    num_labels=4,  # Number of classes in AG News
    dropout=0.5,

    # Training parameters
    batch_size=256,
    learning_rate=1e-3,
    num_epochs=5,

    # Device settings
    use_gpu=torch.cuda.is_available(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
