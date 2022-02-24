import torch
import torch.nn as nn
from rnnt.model.model_base import ModelBase
from rnnt.model.rnn_encoder import RNNEncoder
from rnnt.model.transducer import RNNTransducer
from rnnt.utils import tensor2np, np2tensor, pad_list
from collections import OrderedDict


class ASR(ModelBase):
    def __init__(self, args):
        super(ASR, self).__init__()
        # for decoder
        self.vocab = args.vocab
        self.blank = 0
        self.unk = 1
        self.eos = 2
        self.pad = 3
        
        # Encoder
        self.enc = RNNEncoder(args)
        
        # Decoder
        self.dec = RNNTransducer(args)

        if args.resume != "":
            ckp = torch.load(args.resume,map_location=self.device)
            new_state_dict = OrderedDict()
            for m in ckp:
                if m.startswith("enc."):
                    new_state_dict[m.split('enc.', 1)[1]] = ckp[m]
            self.enc.load_state_dict(new_state_dict)

            new_state_dict = OrderedDict()
            for m in ckp:
                if m.startswith("dec"):
                    new_state_dict[m.split('dec.', 1)[1]] = ckp[m]
            self.dec.load_state_dict(new_state_dict)


    def forward(self, batch, is_eval=False):
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, observation = self._forward(batch)
        else:
            self.train()
            loss, observation = self._forward(batch)

        return loss, observation


    def _forward(self, batch):
        eouts, elens = self.encode(batch['xs'])

        observation = {}
        loss = torch.zeros((1,), dtype=torch.float32, device=self.device)

        loss_fwd, observation = self.dec(eouts, elens, batch['ys'])
        loss += loss_fwd

        return loss, observation


    def encode(self, xs):
        elens = torch.IntTensor([len(x)//4 for x in xs])
        xs = pad_list([np2tensor(x, self.device).float() for x in xs], 0.)

        # encoder
        eouts = self.enc(xs, 100000, False, False)

        return eouts, elens


    def decode_greedy(self, xs):
        eouts, elens = self.encode(xs)

        hyps = self.dec.greedy(eouts, elens)

        return hyps
