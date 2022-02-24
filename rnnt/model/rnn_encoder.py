import torch.nn as nn
import torch
import logging

from rnnt.model.model_base import ModelBase


logger = logging.getLogger(__name__)

class Conv2dBlock(ModelBase):
    def __init__(self, in_channel, out_channel, kernel_size, stride, pooling):
        super(Conv2dBlock, self).__init__()
        self.time_stride = 1
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=tuple(kernel_size), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=tuple(kernel_size), stride=tuple(stride), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=tuple(pooling), stride=tuple(pooling), padding=(0, 0), ceil_mode=True)

    def forward(self, xs : torch.Tensor, 
                lookback : bool = True, 
                lookahead : bool = True) -> torch.Tensor:
        xs = self.conv1(xs)
        xs = torch.relu(xs)
        if lookback and xs.size(2) > self.time_stride:
            xs = xs[:, :, self.time_stride:]
        if lookahead and xs.size(2) > self.time_stride:
            xs = xs[:, :, :xs.size(2) - self.time_stride]

        xs = self.conv2(xs)
        xs = torch.relu(xs)
        if lookback and xs.size(2) > self.time_stride:
            xs = xs[:, :, self.time_stride:]
        if lookahead and xs.size(2) > self.time_stride:
            xs = xs[:, :, :xs.size(2) - self.time_stride]
            
        if self.pool is not None:
            xs = self.pool(xs)
        return xs


class ConvEncoder(ModelBase):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers += [Conv2dBlock(in_channel=1, out_channel=32, kernel_size=(3,3), stride=(1,1), pooling=(2,2))]
        self.layers += [Conv2dBlock(in_channel=32, out_channel=32, kernel_size=(3,3), stride=(1,1), pooling=(2,2))]

    def forward(self, xs : torch.Tensor, 
                lookback : bool = True, 
                lookahead : bool = True) -> torch.Tensor:
        for block in self.layers:
            xs = block(xs, lookback, lookahead)
        return xs


class RNNEncoder(ModelBase):
    def __init__(self, args):
        super(RNNEncoder, self).__init__()
        self.n_layers = args.enc_layers
        self.input_dim = args.enc_input_dim
        self.num_units = args.enc_num_units
        self.subsample_factor = args.subsample_factor
        self.conv = ConvEncoder()
        
        self.rnn = nn.ModuleList()
        self.rnn_bwd = nn.ModuleList()

        for lth in range(self.n_layers):
            if lth == 0:
                self.rnn += [nn.LSTM(self.input_dim, self.num_units, 1, batch_first=True)]
                self.rnn_bwd += [nn.LSTM(self.input_dim, self.num_units, 1, batch_first=True)]
            else:
                self.rnn += [nn.LSTM(self.num_units, self.num_units, 1, batch_first=True)]
                self.rnn_bwd += [nn.LSTM(self.num_units, self.num_units, 1, batch_first=True)]

        self.reset_cache()


    def reset_cache(self):
        self.hx_fwd = [None] * self.n_layers
        logger.debug('Reset cache.')


    def forward(self, 
                xs : torch.Tensor, 
                N_c : int = 40, 
                lookback : bool = True, 
                lookahead : bool = True) -> torch.Tensor:
        self.reset_cache()
        xs = xs.unsqueeze(1)
        xs = self.conv(xs, lookback, lookahead)
        xs = xs.permute(0,2,1,3)
        b = xs.size()[0]
        t = xs.size()[1]
        xs = xs.reshape([b,t,self.input_dim])
        xs_chunks = []
        xs_chunk = xs
        N_c = N_c // self.subsample_factor

        for lth in range(self.n_layers):
            xs_chunk_bwd = torch.flip(self.rnn_bwd[lth](torch.flip(xs_chunk, dims=[1]))[0], dims=[1]) # bwd
            if xs_chunk.size(1) <= N_c: # last chunk
                xs_chunk_fwd, hx=self.hx_fwd[lth] = self.rnn[lth](xs_chunk, hx=self.hx_fwd[lth])
            else:
                xs_chunk_fwd1, self.hx_fwd[lth] = self.rnn[lth](xs_chunk[:, :N_c], hx=self.hx_fwd[lth])
                xs_chunk_fwd2, _ = self.rnn[lth](xs_chunk[:, N_c:], hx=self.hx_fwd[lth])
                xs_chunk_fwd = torch.cat([xs_chunk_fwd1, xs_chunk_fwd2], dim=1)
            xs_chunk = xs_chunk_fwd + xs_chunk_bwd

        xs_chunks.append(xs_chunk[:, :N_c])
        xs = torch.cat(xs_chunks, dim=1)
        return xs