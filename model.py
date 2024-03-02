import torch
import torch.nn as nn
import math
from torch import Tensor
import traceback
import torch.optim as optim
from tqdm.notebook import tqdm
import string
import ast
import pandas as pd
import copy
from sklearn.metrics import precision_recall_fscore_support

device = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = torch.permute(x, (1, 0, 2))
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = torch.permute(x, (1, 0, 2))
        return x

class SpellCorrectionModel(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, char_embedding_dim=256, word_embedding_dim=768, word_n_head=8, char_n_head=4, word_ffw=786, char_ffw=256, dropout=0.5, hidden_dim = 256):
        super(SpellCorrectionModel, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.activation = nn.ReLU()
        self.pos_encoder = PositionalEncoding(word_embedding_dim+char_embedding_dim, dropout=0.1)
        self.norm_w = nn.LayerNorm(word_embedding_dim+char_embedding_dim)
        self.norm_c = nn.LayerNorm(char_embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        word_transformer = nn.TransformerEncoderLayer(d_model=(word_embedding_dim+char_embedding_dim), nhead=word_n_head, dim_feedforward=(word_ffw+char_ffw), dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.word_transformer_encoder = nn.TransformerEncoder(word_transformer, num_layers=8, norm=self.norm_w)
        char_transformer = nn.TransformerEncoderLayer(d_model=char_embedding_dim, nhead=char_n_head, dim_feedforward=char_ffw, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.char_transformer_encoder = nn.TransformerEncoder(char_transformer, num_layers=4, norm=self.norm_c)
        # self.linear_layer = nn.Linear(char_embedding_dim, 768)

        self.correction = nn.Linear(word_embedding_dim+char_embedding_dim, word_vocab_size)
        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.is_correct = nn.Linear(hidden_dim, 1)
        self.is_upper = nn.Linear(hidden_dim, 1)
        self.hidden_correct = nn.Linear(word_embedding_dim+char_embedding_dim, hidden_dim)
        self.hidden_upper = nn.Linear(word_embedding_dim+char_embedding_dim, hidden_dim)

    def forward(self, word_text, char_text):
        w = torch.tensor(word_text).to(device)
        c = torch.tensor(char_text).to(device)

        # print("Input : ",w.shape,c.shape) 
        # assign mask for word and char
        padding_mask_w = (w == 0).float()
        mask = (w == 0)

        length_ = torch.sum((w != 0).float()).item()

        # padding_mask_c = (torch.sum(c,dim=-1,keepdims=False) != 0).unsqueeze(2).float()
        # padding_mask_c = (c != 0).unsqueeze(3).float()

        # print("mask w : ",padding_mask_w.shape)
        # print("mask c : ",padding_mask_c.shape)

        word_embedded_text = self.word_embedding(w)
        char_embedded_text = self.char_embedding(c)

        char_embedded_text = torch.mean(char_embedded_text, dim=2, keepdims=False)
        char_embedded_text = self.char_transformer_encoder(char_embedded_text, src_key_padding_mask=padding_mask_w)
        linear_input = torch.cat((word_embedded_text, char_embedded_text), dim=2)
        src = self.pos_encoder(linear_input)
        output = self.word_transformer_encoder(src, src_key_padding_mask=padding_mask_w)
        
        # Spell
        output_spell = self.hidden_correct(output)
        output_spell = self.activation(output_spell)
        output_spell = self.dropout_layer(output_spell)
        output_spell = self.is_correct(output_spell)
        output_spell = self.sigmoid(output_spell).squeeze()
        
        # Upper
        output_upper = self.hidden_upper(output)
        output_upper = self.activation(output_upper)
        output_upper = self.dropout_layer(output_upper)
        output_upper = self.is_upper(output_upper)
        output_upper = self.sigmoid(output_upper).squeeze()

        # Correction
        output_correction = self.correction(output)


        # print("output ",output_spell) 
        # print("output ",output_correction.shape) 
        # print("output ",output_upper) 
        # print("mask ", mask)
        # print("output ",mask.shape) 
        output_spell.data.masked_fill_(mask, 0)
        # output_correction.data.masked_fill_(mask, 0)
        output_upper.data.masked_fill_(mask, 0)
        
        return output_correction, output_spell, output_upper, length_


    def calculate_metrics(word_text, char_text, label):
        total_words = len(word_text)
        output_correction, output_spell = self.forward(word_text, char_text)
        c = torch.round(output_spell).tolist()
        c_label = label['spell']
        precision, recall, f1_score, _ = precision_recall_fscore_support(c_label, c, average='binary')
        return {"precision": precision, "recall": recall, "f1_score": f1_score}
