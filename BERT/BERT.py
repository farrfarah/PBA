from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from encoder import Encoder

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.init_bert()
        self.encoder = Encoder(opt.enc_method, self.bert_dim, opt.hidden_size, opt.out_size)
        self.cls = nn.Linear(opt.out_size, opt.num_labels)
        self.dropout = nn.Dropout(opt.dropout)

    def init_bert(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.opt.bert_path)
        self.bert = AutoModel.from_pretrained(self.opt.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert_dim = self.opt.bert_dim

    def forward(self, x):
        x = self.bert(input_ids=x, attention_mask=(x > 0))[0]
        x = self.encoder(x)
        x = self.dropout(x)
        return self.cls(x)
