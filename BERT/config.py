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
    bert_dim=768,       # Dimensi embedding
    hidden_size=4,     # Ukuran hidden layer
    out_size=128,       # Ukuran output layer
    num_labels=4,      # Jumlah kelas dalam AG News
    dropout=0.3,       # Tingkat dropout

    # Training parameters
    batch_size=128,     # Batch size untuk training
    learning_rate=5e-4, # Learning rate
    num_epochs=3,       # Jumlah epoch training

    # Device settings
    use_gpu=torch.cuda.is_available(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
