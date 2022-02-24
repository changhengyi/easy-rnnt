import torch.nn as nn
import torch
from collections import OrderedDict
import numpy as np
import logging

import warp_rnnt
from rnnt.utils import *
from rnnt.model.model_base import ModelBase
from rnnt.model.ctc import CTC



logger = logging.getLogger(__name__)

class RNNTransducer(ModelBase):
    def __init__(self, args):
        super(RNNTransducer, self).__init__()
        self.blank = 0
        self.unk = 1
        self.eos = 2
        self.pad = 3

        self.class_num = args.vocab
        
        self.ctc_weight = args.ctc_weight
        self.rnnt_weight = args.rnnt_weight
        self.param_init = args.param_init
        self.n_layers = args.dec_layers
        self.n_units = args.dec_num_units
        self.emb_dim = args.emb_dim
        self.joint_dim = args.joint_dim

        self.embed_cache = None
        self.training = True

        # define network
        self.rnn = nn.ModuleList()
        for lth in range(self.n_layers):
            if lth == 0:
                self.rnn += [nn.LSTM(self.emb_dim, self.n_units, 1, batch_first=True)]
            else:
                self.rnn += [nn.LSTM(self.n_units, self.n_units, 1, batch_first=True)]

        self.embed = nn.Embedding(self.class_num, self.emb_dim, padding_idx=3)
        self.w_enc = nn.Linear(in_features=args.enc_num_units, out_features=self.joint_dim)
        self.w_dec = nn.Linear(in_features=self.n_units, out_features=self.joint_dim, bias=False)
        self.output = nn.Linear(in_features=self.joint_dim, out_features=self.class_num)

        self.dropout = nn.Dropout(p=args.dropout_dec)
        self.dropout_emb = nn.Dropout(p=args.dropout_emb)


        self.ctc = CTC(eos=self.eos,
                           blank=self.blank,
                           enc_n_units=args.enc_num_units,
                           vocab=args.vocab,
                           dropout=args.dropout_dec,
                           lsm_prob=args.ctc_lsm_prob,
                           fc_list=args.ctc_fc_list)


        self.reset_parameters(self.param_init)

        
    def reset_parameters(self, param_init):
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() in [2, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError(n)


    def recurrency(self, ys_emb, dstate):
        """Update prediction network.

        Args:
            ys_emb (FloatTensor): `[B, L, emb_dim]`
            dstate (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`
        Returns:
            dout (FloatTensor): `[B, L, emb_dim]`
            new_dstate (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        if dstate is None:
            dstate = self.zero_state(ys_emb.size(0))
        new_dstate = {'hxs': None, 'cxs': None}

        new_hxs, new_cxs = [], []
        for lth in range(self.n_layers):
            ys_emb, (h, c) = self.rnn[lth](ys_emb, hx=(dstate['hxs'][lth:lth + 1],
                                                       dstate['cxs'][lth:lth + 1]))
            new_hxs.append(h)
            new_cxs.append(c)
            ys_emb = self.dropout(ys_emb)

        # Repackage
        new_dstate['hxs'] = torch.cat(new_hxs, dim=0)
        new_dstate['cxs'] = torch.cat(new_cxs, dim=0)

        return ys_emb, new_dstate


    def zero_state(self, batch_size):
        """Initialize hidden states.

        Args:
            batch_size (int): batch size
        Returns:
            zero_state (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        w = next(self.parameters())
        zero_state = {'hxs': None, 'cxs': None}
        zero_state['hxs'] = w.new_zeros(self.n_layers, batch_size, self.n_units)
        zero_state['cxs'] = w.new_zeros(self.n_layers, batch_size, self.n_units)
        return zero_state


    def embed_token_id(self, indices):
        """Embed token IDs.

        Args:
            indices (LongTensor): `[B]`
        Returns:
            ys_emb (FloatTensor): `[B, vocab, emb_dim]`

        """
        if self.embed_cache is None or self.training:
            ys_emb = self.dropout_emb(self.embed(indices))
        else:
            ys_emb = self.embed_cache[indices]
        return ys_emb


    def joint(self, eouts, douts):
        eouts = eouts.unsqueeze(2)
        douts = douts.unsqueeze(1)
        out = torch.tanh(self.w_enc(eouts) + self.w_dec(douts))
        out = self.output(out)
        return out


    def forward(self, eouts, elens, ys):
        """Forward pass.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
            trigger_points (np.ndarray): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_transducer': None, 'loss_ctc': None, 'loss_mbr': None}
        loss = eouts.new_zeros((1,))

        # CTC loss
        loss_ctc = self.ctc(eouts, elens, ys)
        observation['loss_ctc'] = tensor2scalar(loss_ctc)
        loss += loss_ctc * self.ctc_weight

        # RNN-T loss
        loss_transducer = self.forward_transducer(eouts, elens, ys)
        observation['loss_transducer'] = tensor2scalar(loss_transducer)
        loss += loss_transducer * self.rnnt_weight


        observation['loss'] = tensor2scalar(loss)
        return loss, observation


    def forward_transducer(self, eouts, elens, ys):
        """Compute Transducer loss.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`

        """
        # Append <sos> and <eos>
        _ys = [np2tensor(np.fromiter(y, dtype=np.int64), eouts.device) for y in ys]
        ylens = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
        eos = eouts.new_zeros((1,), dtype=torch.int64).fill_(self.eos)
        ys_in = pad_list([torch.cat([eos, y], dim=0) for y in _ys], self.pad)  # `[B, L+1]`
        ys_out = pad_list(_ys, self.blank)  # `[B, L]`

        # Update prediction network
        dout, _ = self.recurrency(self.embed_token_id(ys_in), None)

        # Compute output distribution
        logits = self.joint(eouts, dout)  # `[B, T, L+1, vocab]`

        # Compute Transducer loss
        log_probs = torch.log_softmax(logits, dim=-1)
        assert log_probs.size(2) == ys_out.size(1) + 1
        ys_out = ys_out.to(eouts.device)
        elens = elens.to(eouts.device)
        ylens = ylens.to(eouts.device)
        
        loss = warp_rnnt.rnnt_loss(log_probs, ys_out.int(), elens, ylens,
                                    average_frames=False,
                                    reduction='mean',
                                    gather=False)
        
        return loss
        

    def greedy(self, eouts, elens):
        """Greedy decoding.
        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
        Returns:
            hyps (List): length `[B]`, each of which contains arrays of size `[L]`
            aw: dummy
        """
        bs = eouts.size(0)

        hyps = []
        for b in range(bs):
            hyp_b = []
            # Initialization
            y = eouts.new_zeros((1, 1), dtype=torch.int64).fill_(self.eos)
            dout, dstate = self.recurrency(self.embed_token_id(y), None)

            for t in range(elens[b]):
                # Pick up 1-best per frame
                out = self.joint(eouts[b:b + 1, t:t + 1], dout)
                y = out.squeeze(2).argmax(-1)
                idx = y[0].item()

                # Update prediction network only when predicting non-blank labels
                if idx != self.blank:
                    hyp_b += [idx]
                    dout, dstate = self.recurrency(self.embed_token_id(y), dstate)

            hyps += [hyp_b]
        return hyps


    def initialize_beam(self, hpy, hxs, cxs):
        hyps = [{'hyp': hpy,
                 'time':[0],
                 'hyp_ids_str': '',
                 'score_rnnt': 0.,
                 'dout': None,
                 'hxs': hxs,
                 'cxs': cxs,
                 'path_len': 0,
                 'update_pred_net': True}]
        return hyps


    def batchfy_pred_net(self, hyps, cache):
        batch_hyps = [beam for beam in hyps if beam['update_pred_net']]
        hyp_ids_strs = [beam['hyp_ids_str'] for beam in hyps]
        for beam in batch_hyps:
            index = hyp_ids_strs.index(beam['hyp_ids_str'])

            y = torch.zeros((1,1), dtype=torch.int64).fill_(beam['hyp'][-1])
            dout, hxs, cxs = self.recurrency(self.embed(y), beam['hxs'], beam['cxs'])

            hyps[index]['update_pred_net'] = False
            hyps[index]['dout'] = dout
            hyps[index]['hxs'] = hxs
            hyps[index]['cxs'] = cxs
            
            cache[beam['hyp_ids_str']] = {
                'dout': dout,
                'hxs': hxs,
                'cxs': cxs
            }
        
        return hyps, cache


    def merge_rnnt_path(self, hyps, merge_prob=False):
        hyps_merged = {}
        for beam in hyps:
            hyp_ids_str = beam['hyp_ids_str']
            if hyp_ids_str not in hyps_merged.keys():
                hyps_merged[hyp_ids_str] = beam
            else:
                if merge_prob:
                    for k in ['score_rnnt']:
                        hyps_merged[hyp_ids_str][k] = np.logaddexp(hyps_merged[hyp_ids_str][k], beam[k])
                    # NOTE: LM scores should not be merged

                elif beam['score_rnnt'] > hyps_merged[hyp_ids_str]['score_rnnt']:
                    # Otherwise, pick up a path having higher log-probability
                    hyps_merged[hyp_ids_str] = beam

        hyps = [v for v in hyps_merged.values()]
        return hyps


    def beam_search(self, eouts, beam_width):
        state_cache = OrderedDict()
        frame_size = eouts.size()[1]
        hxs, cxs = torch.zeros([2, 1, 1024]), torch.zeros([2, 1, 1024])
        hyps = self.initialize_beam([2], hxs, cxs)
        
        for t in range(frame_size):
            hyps, state_cache = self.batchfy_pred_net(hyps, state_cache)
            douts = torch.cat([beam['dout'] for beam in hyps], dim=0)
            logits = self.joint(eouts[:, t:t + 1].repeat([len(hyps), 1, 1]), douts)
            scores_rnnt = torch.log_softmax(logits.squeeze(2).squeeze(1), dim=-1)  # `[B, vocab]`

            new_hyps = []
            for j, beam in enumerate(hyps):
                # Transducer scores
                total_scores_rnnt = beam['score_rnnt'] + scores_rnnt[j]
                total_scores_topk, topk_ids = torch.topk(
                    total_scores_rnnt, k=beam_width, dim=-1, largest=True, sorted=True)

                for k in range(beam_width):
                    idx = topk_ids[k].item()

                    if idx == self.blank:
                        new_hyps.append(beam.copy())
                        new_hyps[-1]['score_rnnt'] += scores_rnnt[j, self.blank].item()
                        new_hyps[-1]['update_pred_net'] = False
                        continue

                    total_score_rnnt = total_scores_topk[k].item()

                    hyp_ids = beam['hyp'] + [idx]
                    hyp_times = beam['time'] + [t*40]
                    hyp_ids_str = ' '.join(list(map(str, hyp_ids)))
                    exist_cache = hyp_ids_str in state_cache.keys()
                    if exist_cache:
                        # from cache
                        dout = state_cache[hyp_ids_str]['dout']
                        hxs = state_cache[hyp_ids_str]['hxs']
                        cxs = state_cache[hyp_ids_str]['cxs']
                    else:
                        # prediction network and LM will be updated later
                        dout = None
                        hxs = beam['hxs']
                        cxs = beam['cxs']

                    new_hyps.append({'hyp': hyp_ids,
                                     'time': hyp_times,
                                     'hyp_ids_str': hyp_ids_str,
                                     'score_rnnt': total_score_rnnt,
                                     'dout': dout,
                                     'hxs': hxs,
                                     'cxs': cxs,
                                     'update_pred_net': not exist_cache})

            # Local pruning
            new_hyps = sorted(new_hyps, key=lambda x: x['score_rnnt'], reverse=True)
            new_hyps = self.merge_rnnt_path(new_hyps, True)
            hpys = new_hyps[:beam_width]

        return 
