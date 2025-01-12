import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: [batch_size, src_len, hid_dim]
        return outputs, hidden, cell  # Return all outputs for attention and hidden states for decoding


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Parameter(torch.rand(hid_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim], encoder_outputs: [batch_size, src_len, hid_dim]
        src_len = encoder_outputs.size(1)

        # Repeat hidden for each time step in src_len
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hid_dim]

        # Concatenate hidden and encoder_outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]
        energy = torch.matmul(energy, self.v.unsqueeze(0).transpose(0, 1))  # [batch_size, src_len, 1]
        attn_weights = torch.softmax(energy.squeeze(2), dim=1)  # [batch_size, src_len]

        return attn_weights  # [batch_size, src_len]


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, attention, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(hid_dim + emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, hidden, cell, encoder_outputs):
        # trg: [batch_size], hidden: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        trg = trg.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(trg))  # [batch_size, 1, emb_dim]

        # Compute attention weights
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]

        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hid_dim]

        # Combine embedded and context
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + hid_dim]

        # Decode the current token
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [batch_size, 1, hid_dim]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]

        return prediction, hidden, cell  # Return prediction and updated states