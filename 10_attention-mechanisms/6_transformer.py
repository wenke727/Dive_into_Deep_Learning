#%%
import os
import math
from zmq import device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch import dropout, nn
from torch.nn import functional as F
from d2l import torch as d2l

%matplotlib inline

# %%
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        # TODO LayerBorm vs BatchNorm
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, X, Y):
        return self.ln(self.dropout(Y)+X)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.ffn       = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnomal  = AddNorm(norm_shape, dropout)
        self.addnorm2  = AddNorm(norm_shape, dropout)
        
    def forward(self, X, valid_lens):
        Y = self.addnomal(X, self.attention(X, X, X, valid_lens))
        
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    """transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size, 
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens  = num_hiddens
        self.embedding    = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks         = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module(
                "block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias)
            )
    
    def forward(self, X, valid_lens, *args):
        # 此嵌⼊值乘以嵌⼊维度的平⽅根进⾏缩放
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        
        return X
        

class DecoderBlock(nn.Module):
    """解码器中第 i 个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.addnorm3 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        
        # state[2][self.i]包含着直到当前时间步第i个块解码的输出表⽰
        if state[2][self.i] is None:
            key_values = X
        else:
            # TODO 加起来的含义是？<- 经测试, `推理` 时调用
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device)\
                                  .repeat(batch_size, 1)
        else:
            dec_valid_lens = None
            
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        
        return self.addnorm3(Z, self.ffn(Z)), state
    

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens  = num_hiddens
        self.num_layers   = num_layers
        self.embedding    = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks         = nn.Sequential()
        self.dense        = nn.Linear(num_hiddens, vocab_size)

        for i in range(num_layers):
            self.blks.add_module(
                'block' + str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens, 
                             num_heads, dropout, i)
            )
    
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens) )
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        
        return self.dense(X), state
    
    @property
    def attention_weights(self):
        return self._attention_weights


class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)

        return self.decoder(dec_X, dec_state)


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence.

    Defined in :numref:`sec_seq2seq_training`"""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


""" training """
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, .1, 64, 10
lr, num_epochs, device = .005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout
)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout
)
net = d2l.EncoderDecoder(encoder, decoder)

d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# just for a check
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')


#%%
def check():
    """ checking """
    # PositionWiseFFN
    ffn = PositionWiseFFN(4, 4, 8)
    ffn.eval()
    ffn(torch.ones(2, 3, 4)).shape
            

    # nn.LayerNorm
    ln = nn.LayerNorm(2)
    bn = nn.BatchNorm1d(2)
    X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    print('layer norm: ',ln(X), '\nbatch norm:',bn(X))

    
    # Encoder
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    # TODO norm_shape = [100, 24] ???
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, .5)
    encoder_blk.eval()
    encoder_blk(X, valid_lens).shape


    # TransformerEncoder
    encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, .5)
    encoder.eval()
    encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape


    # EncoderBlock
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    encoder_blk(X, valid_lens).shape




# %%
""" 可视化权重 """
enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads, -1, num_steps))
enc_attention_weights.shape, type(net.encoder.attention_weights)

# %%
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))

# %%
import pandas as pd
dec_attention_weight_2d = [
    head[0].tolist()
    for step in dec_attention_weight_seq
        for attn in step
            for blk in attn
                for head in blk
]
dec_attention_weight_filled = torch.tensor(
    pd.DataFrame(dec_attention_weight_2d).fillna(.0).values
)
dec_attention_weights = dec_attention_weight_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
dec_self_attention_weights.shape, dec_inter_attention_weights.shape

#%%
# Plus one to include the beginning-of-sequence token
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

# %%
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))

# %%
