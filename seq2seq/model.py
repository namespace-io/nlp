import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        #src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        #embedded = [src_len, batch_size, emb_dim]

        outputs, hidden = self.rnn(embedded)
        #outputs = [src_len, batch_size, num_direction * hid_dim]
        #hidden  = [n_layer * num_direction, batch_size, hid_dim]

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        #outputs = [src_len, batch_size, num_direction * hid_dim]
        #hidden  = [batch_size, dec_hid_dim]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        #hidden = [batch_size, dec_hid_dim]
        #encoder_outputs = [src_len, batch_size, enc_hid_dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #hidden = [batch_size, src_len, dec_hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch_size, src_len, enc_hid_dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        #energy = [batch_size,  src_len, dec_hid_dim]
        energy = energy.permute(0, 2, 1)
        #energy = [batch_size, dec_hid_dim, src_len]
        #v = [dec_hid_dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        #v = [batch_size, 1, dec_hid_dim]

        attention = torch.bmm(v, energy).squeeze(1)

        #attention = [batch_size, src_len]

        return F.softmax(attention, dim = 1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        #input = [batch_size]
        #hidden = [batch_size, dec_hid_dim]
        #encoder_outputs = [src_len, batch_size, end_hid_dim * 2]

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch_size, emb_dim]
        a = self.attention(hidden, encoder_outputs)
        #a = [batch_size, src_len]
       
        a = a.unsqueeze(1)
        #a = [batch_size, 1, src_len]
       
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
       
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch_size, enc_hid_dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output  = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        #prediction = [batch_size, output_size]
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src_len, batch_size]
        #trg = [trg_len, batch_size]

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        #first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[t] if teacher_force else top1
        
        return outputs

