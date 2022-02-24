import torch.nn as nn
import torch
from collections import OrderedDict
from rnnt.utils import *
from rnnt.model.model_base import ModelBase
import math


class CTC(ModelBase):
    def __init__(self,
                 eos,
                 blank,
                 enc_n_units,
                 vocab,
                 dropout=0.,
                 lsm_prob=0.,
                 fc_list=None,
                 backward=False):

        super(CTC, self).__init__()

        self.eos = eos
        self.blank = blank
        self.vocab = vocab
        self.lsm_prob = lsm_prob
        self.bwd = backward

        self.space = -1  # TODO(hirofumi): fix later

        # for cache
        self.prev_spk = ''
        self.lmstate_final = None

        # for posterior plot
        self.prob_dict = {}
        self.data_dict = {}

        # Fully-connected layers before the softmax
        if fc_list is not None and len(fc_list) > 0:
            _fc_list = [int(fc) for fc in fc_list.split('_')]
            fc_layers = OrderedDict()
            for i in range(len(_fc_list)):
                input_dim = enc_n_units if i == 0 else _fc_list[i - 1]
                fc_layers['fc' + str(i)] = nn.Linear(input_dim, _fc_list[i])
                fc_layers['dropout' + str(i)] = nn.Dropout(p=dropout)
            fc_layers['fc' + str(len(_fc_list))] = nn.Linear(_fc_list[-1], vocab)
            self.output = nn.Sequential(fc_layers)
        else:
            self.output = nn.Linear(enc_n_units, vocab)
        
        self.ctc_loss = nn.CTCLoss(reduction="sum", zero_infinity=True)


    def forward(self, eouts, elens, ys):
        """Compute CTC loss.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (List): length `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`
            trigger_points (IntTensor): `[B, L]`

        """
        # Concatenate all elements in ys for warpctc_pytorch
        ylens = np2tensor(np.fromiter([len(y) for y in ys], dtype=np.int32))
        ys_ctc = torch.cat([np2tensor(np.fromiter(y[::-1] if self.bwd else y, dtype=np.int32))
                            for y in ys], dim=0)
        # NOTE: do not copy to GPUs here

        # Compute CTC loss
        logits = self.output(eouts)
        loss = self.loss_fn(logits.transpose(1, 0), ys_ctc, elens, ylens)

        # Label smoothing for CTC
        if self.lsm_prob > 0:
            loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(logits, elens) * self.lsm_prob


        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.prob_dict['probs'] = tensor2np(torch.softmax(logits, dim=-1))

        return loss


    def loss_fn(self, logits, ys_ctc, elens, ylens):
        # Use the deterministic CuDNN implementation of CTC loss to avoid
        #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
        with torch.backends.cudnn.flags(deterministic=True):
            loss = self.ctc_loss(logits.log_softmax(2), ys_ctc, elens, ylens) / logits.size(1)
        return loss


def kldiv_lsm_ctc(logits, ylens):
    """Compute KL divergence loss for label smoothing of CTC and Transducer models.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ylens (IntTensor): `[B]`
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs, _, vocab = logits.size()

    log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = torch.mul(probs, log_probs - log_uniform)
    loss_mean = sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
    return loss_mean