import torch.nn as nn
import torch
from collections import OrderedDict
import numpy as np
import logging

import warp_rnnt
from rnnt.utils import *
from rnnt.model.model_base import ModelBase
from rnnt.model.ctc import CTC
from rnnt.model.lm import NgramLM


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

        self.lm = NgramLM()

        
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
            if self.training:
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
        if self.training:
            ys_emb = self.dropout_emb(self.embed(indices))
        elif self.embed_cache is None:
            ys_emb = self.embed(indices)
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
        self.training = False
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
        self.training = True
        return hyps


    def initialize_beam(self, hpy, dstate, lmstate):
        hyps = [{'hyp': hpy,
                 'time':[0],
                 'hyp_ids_str': '',
                 'score':0.,
                 'score_rnnt': 0.,
                 'score_lm': 0.,
                 'dout': None,
                 'dstate': dstate,
                 'lmstate': lmstate,
                 'words_list' : [],
                 'path_len': 0,
                 'update_pred_net': True}]
        return hyps


    def batchfy_pred_net(self, hyps, cache):
        batch_hyps = [beam for beam in hyps if beam['update_pred_net']]
        if len(batch_hyps) == 0:
            return hyps, cache

        ys = torch.zeros((len(batch_hyps), 1), dtype=torch.int64, device=self.device)
        for i, beam in enumerate(batch_hyps):
            ys[i] = beam['hyp'][-1]
        dstates_prev = {'hxs': torch.cat([beam['dstate']['hxs'] for beam in batch_hyps], dim=1),
                        'cxs': torch.cat([beam['dstate']['cxs'] for beam in batch_hyps], dim=1)}
        douts, dstates = self.recurrency(self.embed_token_id(ys), dstates_prev)

        hyp_ids_and_words_strs = [beam['hyp_ids_str'] + "".join(beam['words_list']) for beam in hyps]
        for i, beam in enumerate(batch_hyps):
            dstate = {'hxs': dstates['hxs'][:, i:i + 1],
                      'cxs': dstates['cxs'][:, i:i + 1]}
            index = hyp_ids_and_words_strs.index(beam['hyp_ids_str'] + "".join(beam['words_list']))

            hyps[index]['dout'] = douts[i:i + 1]
            hyps[index]['dstate'] = dstate
            assert hyps[index]['update_pred_net']
            hyps[index]['update_pred_net'] = False

            # register to cache
            cache[beam['hyp_ids_str']] = {
                'dout': douts[i:i + 1],
                'dstate': dstate
            }
        
        return hyps, cache


    def merge_rnnt_path(self, hyps, merge_prob=False):
        hyps_merged = {}
        for beam in hyps:
            hyp_ids_and_words_str = beam['hyp_ids_str'] + "".join(beam['words_list'])
            if hyp_ids_and_words_str not in hyps_merged.keys():
                hyps_merged[hyp_ids_and_words_str] = beam
            else:
                if merge_prob:
                    for k in ['score_rnnt']:
                        hyps_merged[hyp_ids_and_words_str][k] = np.logaddexp(hyps_merged[hyp_ids_and_words_str][k], beam[k])
                    # NOTE: LM scores should not be merged

                elif beam['score'] > hyps_merged[hyp_ids_and_words_str]['score']:
                    # Otherwise, pick up a path having higher log-probability
                    hyps_merged[hyp_ids_and_words_str] = beam

        hyps = [v for v in hyps_merged.values()]
        return hyps


    def beam_search(self, eouts, elens, beam_width, lm_weight = 0.2):
        self.training = False
        bs = eouts.size(0)
        hyps = []
        for b in range(bs):
            state_cache = OrderedDict()
            hxs, cxs = eouts.new_zeros(2, 1, 1024), eouts.new_zeros(2, 1, 1024)

            first_lm_state = self.lm.get_first_state()

            hyps_b = self.initialize_beam([2], {"hxs":hxs, "cxs":cxs}, first_lm_state)
            eout = eouts[b:b + 1, :elens[b]]
            frame_size = eout.size(1)
            
            for t in range(frame_size):
                hyps_b, state_cache = self.batchfy_pred_net(hyps_b, state_cache)
                douts = torch.cat([beam['dout'] for beam in hyps_b], dim=0)
                logits = self.joint(eout[:, t:t + 1].repeat([len(hyps_b), 1, 1]), douts)
                
                scores_rnnt = torch.log_softmax(logits.squeeze(2).squeeze(1), dim=-1)  # `[B, vocab]`

                new_hyps = []
                for j, beam in enumerate(hyps_b):
                    # Transducer scores
                    total_scores_rnnt = beam['score_rnnt'] + scores_rnnt[j]
                    total_scores_topk, topk_ids = torch.topk(total_scores_rnnt, k=beam_width, dim=-1, largest=True, sorted=True)

                    words_lists = self.lm.pinyins2words(total_scores_topk, topk_ids, beam_width, beam['lmstate'], beam['score_lm'], lm_weight)
                    
                    for k in range(len(words_lists)):
                        idx = words_lists[k][1]
                        if idx == self.blank:
                            new_hyps.append(beam.copy())
                            new_hyps[-1]['score'] += scores_rnnt[j, self.blank].item()
                            new_hyps[-1]['score_rnnt'] += scores_rnnt[j, self.blank].item()
                            new_hyps[-1]['update_pred_net'] = False
                            continue

                        total_score_rnnt = words_lists[k][2]
                        total_score_lm = words_lists[k][3]
                        total_score = total_score_rnnt + total_score_lm * lm_weight

                        hyp_ids = beam['hyp'] + [idx]
                        hyp_times = beam['time'] + [t*40]
                        hyp_ids_str = ' '.join(list(map(str, hyp_ids)))
                        exist_cache = hyp_ids_str in state_cache.keys()
                        lmstate = self.lm.update(beam['lmstate'], words_lists[k][0])
                        words_list = beam['words_list'] + [words_lists[k][0]]

                        if exist_cache:
                            # from cache
                            dout = state_cache[hyp_ids_str]['dout']
                            dstate = state_cache[hyp_ids_str]['dstate']
                        else:
                            # prediction network will be updated later
                            dout = None
                            dstate = beam['dstate']

                        new_hyps.append({'hyp': hyp_ids,
                                        'time': hyp_times,
                                        'hyp_ids_str': hyp_ids_str,
                                        'score': total_score,
                                        'score_rnnt': total_score_rnnt,
                                        'score_lm':total_score_lm,
                                        'dout': dout,
                                        'dstate': dstate,
                                        'lmstate': lmstate,
                                        'words_list' : words_list,
                                        'update_pred_net': not exist_cache})

                new_hyps = sorted(new_hyps, key=lambda x: x['score'], reverse=True)
                new_hyps = self.merge_rnnt_path(new_hyps, False)
                hyps_b = new_hyps[:beam_width]
                
                for hyp in hyps_b:
                    logger.debug(hyp['words_list'], "total:", hyp['score'], "rnnt:", hyp['score_rnnt'], "lm:", hyp['score_lm'])

            hyps += [hyps_b]
        
        self.training = True

        return hyps