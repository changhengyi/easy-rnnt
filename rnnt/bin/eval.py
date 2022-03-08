import sys
sys.path.insert(0,"/home/changhengyi/easy-rnnt")
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
import os
from tqdm import tqdm

from rnnt.utils import hyp2text_char, compute_wer, hyp2text_wp
from rnnt.dataset.build import build_dataloader
from rnnt.model.asr_model import ASR
from rnnt.bin.args import get_conf

def test01(args):
    ### init data loader
    test_loader = build_dataloader(args, args.testset_path, 1, num_workers=args.num_workers)

    ### init model
    model = ASR(args)
    model.cuda()
    eval_testset(args, model, test_loader, "beamsearch")


def eval_testset(args, model, dataloader, decode_mode="greedy"):
    # load dict to make id2token
    id2token = {0: "<blank>"}
    with open(args.dict, "r") as f:
        for line in f.readlines():
            token, idx = line.strip().split(" ")
            idx = int(eval(idx))
            id2token[idx] = token
    
    pbar = tqdm(total=len(dataloader), ncols=100)
    total, cnt =len(dataloader), 0
    wer, n_sub_w, n_ins_w, n_del_w, n_word = 0, 0, 0, 0, 0
    for batch in dataloader:
        if decode_mode == "beamsearch":
            hyps = model.decode_beam_search(batch['xs'], args.beam_size)
        elif decode_mode == "greedy":
            hyps = model.decode_greedy(batch['xs'])
        else:
            raise NotImplementedError

        assert len(hyps) == len(batch['text'])
        for i in range(len(hyps)):
            ref = batch['text'][i]
            if decode_mode == "beamsearch":
                err_b, sub_b, ins_b, del_b = 1000000, 1000000, 1000000, 1000000
                for h in hyps[i]:
                    hyp = hyp2text_char(h['hyp'][1:], id2token)
                    e_b, s_b, i_b, d_b = compute_wer(ref.split(" "), hyp.split(" "))
                    if e_b <= err_b:
                        err_b, sub_b, ins_b, del_b = e_b, s_b, i_b, d_b
            elif decode_mode == "greedy":
                hyp = hyp2text_char(hyps[i], id2token)
                err_b, sub_b, ins_b, del_b = compute_wer(ref.split(" "), hyp.split(" "))
            else:
                raise NotImplementedError

            #print("{} | ref: {}".format(batch['utt_ids'][i], ref))
            #print("{} | hyp: {}".format(batch['utt_ids'][i], hyp))
            
            wer += err_b
            n_sub_w += sub_b
            n_ins_w += ins_b
            n_del_w += del_b
            n_word += len(ref.split(' '))

        pbar.update(len(batch['utt_ids']))
        # pbar.set_description("[WER : {:.2f}%]".format(wer/n_word))

    wer /= n_word
    n_sub_w /= n_word
    n_ins_w /= n_word
    n_del_w /= n_word
    
    with open(args.log_path, "a") as fw:
        fw.write('Dataset: {}\t| Num: {}\t| ACC: {:.2f}%\t| WER: {:.2f}%\t| SUB: {:.2f}\t| INS: {:.2f}\t| DEL: {:.2f}\n'.format(dataloader.name, len(dataloader),100-wer, wer, n_sub_w, n_ins_w, n_del_w))
    dataloader.reset(is_new_epoch=True)
    pbar.close()


if __name__ == "__main__":
    args = get_conf(OmegaConf.from_cli())
    test01(args)