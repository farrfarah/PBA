from config import config
from preprocessing import preprocess_data
from dataloader import create_dataloader
from BERT import Model
from train import train_model
from test import test_model
import os
import torch

def main():
    # Initialize model and tokenizer
    model = Model(config)
    tokenizer = model.tokenizer

    # Preprocess train and test data
    train_texts, train_labels = preprocess_data(config.train_data_path, tokenizer)
    test_texts, test_labels = preprocess_data(config.test_data_path, tokenizer)

    # Create dataloaders
    train_loader = create_dataloader(train_texts, train_labels, config.batch_size)
    test_loader = create_dataloader(test_texts, test_labels, config.batch_size)

    # Train model
    print("Training model...")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    train_model(model, train_loader, config)
    torch.save(model.state_dict(), config.checkpoint_dir)

    # Test model
    print("Testing model...")
    model.load_state_dict(torch.load(config.checkpoint_dir))
    test_model(model, test_loader, config)

if __name__ == "__main__":
    main()
