import torch
import nn
import math
class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * ((self.d_model)**0.5) 

class PositionalEmbedding(nn.Module):

    def __init__(self ,d_model, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #creating a matrix of size (seq_lem, d_model)
        pe = torch.zeros(seq_len, d_model)
        #create a vector of shape (seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0) / d_model))
        #applying sine to even positions & cosine to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)

class LayerNorm(nn.Module):

    def __init__(self, eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #mul
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim= -1, keepdim = True)
        std = x.std(dim= -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

    


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self,x):
        # (Batch, Seq_Len, d_model) --> 1st layer --> (Batch, seq_len, d_ff) --> 2nd layer --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 

    

    