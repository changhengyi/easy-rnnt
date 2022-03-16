import os
import kenlm
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseLM:
    def __init__(self):
        print("Overiding BaseLM")

    def get_first_state(self):
        raise NotImplementedError

    def update(self, state, word):
        raise NotImplementedError

    def get_word_score(self, state, word):
        raise NotImplementedError

    def get_sentence_score(self, sentence):
        raise NotImplementedError


class PinyinDict:
    def __init__(self):
        self.pinyin2word = {}
        with open("/home/changhengyi/rnnt-decode_test/pinyin.txt", "r") as f:
            for line in f.readlines():
                pinyin, words = line.split("\t")
                self.pinyin2word[pinyin] = words.split()

        self.idx2pinyin = {}
        # load dict to make id2token
        self.id2token = {0: "<blank>"}
        with open("/home/changhengyi/rnnt-decode_test/dict.txt", "r") as f:
            for line in f.readlines():
                token, idx = line.strip().split(" ")
                idx = int(eval(idx))
                self.id2token[idx] = token

    def pinyin2words(self, pinyin_idx):
        pinyin = self.id2token[pinyin_idx]
        if pinyin in self.pinyin2word:
            return self.pinyin2word[pinyin]
        else:
            return [pinyin]


class NgramLM(BaseLM):
    def __init__(self, arpa_file="/ai-cephfs/Share/wwt_share/LM/all_text.binary"):
        self.pinyin_dict = PinyinDict()
        self.model = kenlm.LanguageModel(arpa_file)

    def get_first_state(self):
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        return state

    def update(self, state, word):
        cur_state = kenlm.State()
        self.model.BaseScore(state, word, cur_state)
        return cur_state

    def get_word_score(self, state, word):
        if word == "<blank>":
            return 0
        cur_state = kenlm.State()
        score = self.model.BaseScore(state, word, cur_state)
        return score

    def get_sentence_score(self, sentence):
        words = list(sentence.strip())
        seg_sentence = ' '.join(words)
        score = self.model.score(seg_sentence, eos=False)
        return score

    def pinyins2words(self, total_scores_topk, topk_ids, k, pre_state, pre_lm_score, lm_weight):
        total_scores_topk, topk_ids = total_scores_topk.detach().cpu().numpy(), topk_ids.detach().cpu().numpy()

        '''words_dict : {对应汉字: [拼音索引, 声学得分, 语言得分], ...}'''
        words_dict = {}

        blank_idx = k
        for i in range(k):
            if topk_ids[i] == 0:
                blank_idx = i
                continue
            words = self.pinyin_dict.pinyin2words(topk_ids[i])
            for word in words:
                if word in words_dict:
                    if total_scores_topk[i] > words_dict[word][1]:
                        words_dict[word][0] = topk_ids[i]
                        words_dict[word][1] = total_scores_topk[i] # a little problem here FIXME
                else:
                    word_score = pre_lm_score + self.get_word_score(pre_state, word)
                    words_dict[word] = [topk_ids[i], total_scores_topk[i], word_score]
        
        '''words_list: [[对应汉字, 拼音索引, 声学得分, 语言得分], ...]'''
        words_list = [[word] + words_dict[word] for word in words_dict]
        
        words_list = sorted(words_list, key=lambda x: x[2]+x[3], reverse=True)

        if blank_idx < k:
            words_list = words_list[:blank_idx] + [['blank', 0, total_scores_topk[blank_idx], pre_lm_score]] + words_list[blank_idx+1:]
        words_list = words_list[:k]
        
        logger.debug(words_list)

        return words_list