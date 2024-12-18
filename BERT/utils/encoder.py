import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, method, input_dim, hidden_size, out_size):
        super(Encoder, self).__init__()
        if method == "rnn":
            self.encoder = nn.RNN(input_dim, hidden_size, batch_first=True)
        else:
            raise ValueError("Unsupported encoder method")
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        _, hidden = self.encoder(x)
        return self.fc(hidden.squeeze(0))
