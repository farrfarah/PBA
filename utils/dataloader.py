from torch.utils.data import Dataset, DataLoader
import torch

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

def create_dataloader(texts, labels, batch_size):
    dataset = TextDataset(texts, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
