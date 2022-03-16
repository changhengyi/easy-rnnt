import sys

import numpy as np
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
from utils.lm_tree import LMTree

def test01(args):
    ### init data loader
    test_loader = build_dataloader(args, args.testset_path, 1, args.cmvn_path, num_workers=args.num_workers)
    # for batch in tqdm(test_loader):
    #     utt_id = batch['utt_ids'][0]
    #     save_path = "/home/changhengyi/rnnt-decode_test/feats_cmvn/{}.npy".format(utt_id)
    #     np.save(save_path, batch['xs']) 
    # return 

    ### init model
    model = ASR(args)
    model.cuda()
    eval_testset(args, model, test_loader, "beamsearch_word")


def test02(args):
    ### init data loader
    test_loader = build_dataloader(args, args.testset_path, 1, args.cmvn_path, num_workers=args.num_workers)
    eval_lm(args, test_loader)


def eval_lm(args, dataloader):
    hyp_file = open("{}/{}.wer".format(args.log_path, dataloader.name), "a")
    # load dict to make id2token
    arpa_file = "/ai-cephfs/Share/hyc_share/lm/trigram.binary"
    dict_path = "/ai-cephfs/Share/hyc_share/lm/pinyin2words.txt"
    lm_tree = LMTree(arpa_file, dict_path)
    
    pbar = tqdm(total=len(dataloader), ncols=100)
    wer, n_sub_w, n_ins_w, n_del_w, n_word = 0, 0, 0, 0, 0
    for batch in dataloader:
        for i in range(len(batch['text'])):  
            ref = batch['textorg'][i]
            words_list = lm_tree.translate_whole_sentence(batch['text'][i])
            hyp = "".join(words_list)
            err_b, sub_b, ins_b, del_b = compute_wer(ref, hyp)
            

            hyp_file.write("-------------------------------\n{}\n".format(batch['utt_ids'][i]))
            hyp_file.write("ref: {}\n".format(ref))
            hyp_file.write("hyp: {}\n".format(hyp))
            hyp_file.write("pinyin: {}\n".format(batch['text'][i]))
            hyp_file.write("word: {}\n".format(" ".join(words_list)))
            hyp_file.write("WER: {:.2f}%\t| SUB: {:.2f}\t| INS: {:.2f}\t| DEL: {:.2f}\n".format(err_b/len(ref), sub_b/len(ref), ins_b/len(ref), del_b/len(ref)))
            
            wer += err_b
            n_sub_w += sub_b
            n_ins_w += ins_b
            n_del_w += del_b
            n_word += len(ref)

        pbar.update(len(batch['utt_ids']))
        pbar.set_description("[WER : {:.2f}%]".format(wer/n_word))

    wer /= n_word
    n_sub_w /= n_word
    n_ins_w /= n_word
    n_del_w /= n_word
    # print('Dataset: {}\t| Num: {}\t| ACC: {:.2f}%\t| WER: {:.2f}%\t| SUB: {:.2f}\t| INS: {:.2f}\t| DEL: {:.2f}\n'.format(dataloader.name, len(dataloader),100-wer, wer, n_sub_w, n_ins_w, n_del_w))
    final_result_str = 'Dataset: {}\t| Num: {}\t| ACC: {:.2f}%\t| WER: {:.2f}%\t| SUB: {:.2f}\t| INS: {:.2f}\t| DEL: {:.2f}\n' \
                                       .format(dataloader.name, len(dataloader),100-wer, wer, n_sub_w, n_ins_w, n_del_w)

    hyp_file.write(final_result_str)
    with open(args.log_path + "/final.res", "a") as fw:
        fw.write(final_result_str)
    dataloader.reset(is_new_epoch=True)
    pbar.close()

    hyp_file.close()


def eval_testset(args, model, dataloader, decode_mode="greedy"):
    hyp_file = open("{}/{}.wer".format(args.log_path, dataloader.name), "a")
    # load dict to make id2token
    id2token = {0: "<blank>"}
    with open(args.dict, "r") as f:
        for line in f.readlines():
            token, idx = line.strip().split(" ")
            idx = int(eval(idx))
            id2token[idx] = token
    
    pbar = tqdm(total=len(dataloader), ncols=100)
    wer, n_sub_w, n_ins_w, n_del_w, n_word = 0, 0, 0, 0, 0
    for batch in dataloader:
        if decode_mode.startswith("beamsearch"):
            hyps = model.decode_beam_search(batch['xs'], args.beam_size)
        elif decode_mode == "greedy":
            hyps = model.decode_greedy(batch['xs'])
        else:
            raise NotImplementedError

        assert len(hyps) == len(batch['text'])
        for i in range(len(hyps)):
            if decode_mode == "beamsearch_pinyin":
                ref = batch['text'][i]
                err_b, sub_b, ins_b, del_b = 1000000, 1000000, 1000000, 1000000
                for h in hyps[i]:
                    hyp = hyp2text_char(h['hyp'][1:], id2token)
                    e_b, s_b, i_b, d_b = compute_wer(ref.split(" "), hyp.split(" "))
                    if e_b <= err_b:
                        err_b, sub_b, ins_b, del_b = e_b, s_b, i_b, d_b
            elif decode_mode == "beamsearch_word":
                ref = batch['textorg'][i]
                hyp = "".join(hyps[i][0]['words_list'])
                err_b, sub_b, ins_b, del_b = compute_wer(ref, hyp)
            elif decode_mode == "greedy":
                ref = batch['text'][i]
                hyp = hyp2text_char(hyps[i], id2token)
                err_b, sub_b, ins_b, del_b = compute_wer(ref.split(" "), hyp.split(" "))
            else:
                raise NotImplementedError

            hyp_file.write("-------------------------------\n{}\n".format(batch['utt_ids'][i]))
            hyp_file.write("ref: {}\n".format(ref))
            hyp_file.write("hyp: {}\n".format(hyp))
            hyp_file.write("WER: {:.2f}%\t| SUB: {:.2f}\t| INS: {:.2f}\t| DEL: {:.2f}\n".format(err_b/len(ref), sub_b/len(ref), ins_b/len(ref), del_b/len(ref)))
            
            wer += err_b
            n_sub_w += sub_b
            n_ins_w += ins_b
            n_del_w += del_b
            n_word += len(ref)

        pbar.update(len(batch['utt_ids']))
        pbar.set_description("[WER : {:.2f}%]".format(wer/n_word))

    wer /= n_word
    n_sub_w /= n_word
    n_ins_w /= n_word
    n_del_w /= n_word
    # print('Dataset: {}\t| Num: {}\t| ACC: {:.2f}%\t| WER: {:.2f}%\t| SUB: {:.2f}\t| INS: {:.2f}\t| DEL: {:.2f}\n'.format(dataloader.name, len(dataloader),100-wer, wer, n_sub_w, n_ins_w, n_del_w))
    final_result_str = 'Dataset: {}\t| Num: {}\t| ACC: {:.2f}%\t| WER: {:.2f}%\t| SUB: {:.2f}\t| INS: {:.2f}\t| DEL: {:.2f}\n' \
                                       .format(dataloader.name, len(dataloader),100-wer, wer, n_sub_w, n_ins_w, n_del_w)

    hyp_file.write(final_result_str)
    with open(args.log_path + "/final.res", "a") as fw:
        fw.write(final_result_str)
    dataloader.reset(is_new_epoch=True)
    pbar.close()

    hyp_file.close()


def eval_single_sentence(args):
    # load dict to make id2token
    id2token = {0: "<blank>"}
    with open(args.dict, "r") as f:
        for line in f.readlines():
            token, idx = line.strip().split(" ")
            idx = int(eval(idx))
            id2token[idx] = token
    
    utt_id, ground_truth = "M3R9aCo3vYE____196", "三届世界杯十二个春秋十粒进球以一次十六强"
    # utt_id, ground_truth = "wS9yC0OofCs__20190815_CCTV_2", "晚上好"
    xs = np.load("/home/changhengyi/rnnt-decode_test/feats_cmvn/{}.npy".format(utt_id))

    ### init model
    model = ASR(args)
    model.cuda()
    hyps = model.decode_beam_search(xs, args.beam_size)

    # print(hyps[0][0])

    for h in hyps[0]:
        hyp = hyp2text_char(h['hyp'][1:], id2token)
        print("".join(h['words_list']), hyp)
    
    print("-"*40)
    print("Text:", ground_truth)


if __name__ == "__main__":
    args = get_conf(OmegaConf.from_cli())
    # args.resume="/home/changhengyi/rnnt-decode_test/model.epoch-28"
    # args.beam_size=3
    # eval_single_sentence(args)
    test02(args)