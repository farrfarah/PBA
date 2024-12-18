# -*- coding: utf-8 -*-
"""lstm

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B30BUFkyeNkxoISNkcp-IM0IvbvsgXcG
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    def __init__(
        self,
        n_classes: int,
        vocab_size: int,
        embeddings: torch.Tensor,
        emb_size: int,
        fine_tune: bool,
        rnn_size: int,
        rnn_layers: int,
        dropout: float
    ) -> None:
        super(BiLSTM, self).__init__()

        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)

        # bidirectional LSTM
        self.BiLSTM = nn.LSTM(
            emb_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=(0 if rnn_layers == 1 else dropout),
            batch_first=True
        )

        # fully connected layer
        self.fc = nn.Linear(2 * rnn_size, n_classes)

        self.dropout = nn.Dropout(dropout)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool = True) -> None:
        """
        Set weights for embedding layer
        """
        if embeddings is None:
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) -> torch.Tensor:
        # word embedding, apply dropout
        embeddings = self.dropout(self.embeddings(text))

        # pack sequences (remove word-pads, SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(
            embeddings,
            lengths=words_per_sentence.tolist(),
            batch_first=True,
            enforce_sorted=False
        )

        rnn_out, _ = self.BiLSTM(packed_words)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        H = torch.mean(rnn_out, dim=1)
        scores = self.fc(self.dropout(H))

        return scores