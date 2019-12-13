from model import *


def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ffn=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
    position = PositionalEncoding(d_model, dropout)
    transformer = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ffn), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ffn), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

transformer = make_model(10, 10, 2)