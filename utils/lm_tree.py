import copy
import kenlm


class Path:
    def __init__(self, prefix, pinyin_history, word_history, lm_score, lm_state):
        self.prefix = prefix
        self.pinyin_history = pinyin_history
        self.word_history = word_history
        self.lm_score = lm_score
        self.lm_state = lm_state
        self.is_valid = True

    def __str__(self) -> str:
        pass


class LMTree:
    def __init__(self, arpa_file, dict_path):
        self.max_match_word_len = 7
        self.prune_size = 10

        self.lm_model = kenlm.LanguageModel(arpa_file)
        self.words_dict = self.get_words_dict(dict_path)
        self.paths = [Path([], [], [], 0, self.get_first_state())]

    def reset(self):
        self.paths = [Path([], [], [], 0, self.get_first_state())]
    
    def get_words_dict(self, dict_path):
        words_dict = {}
        with open(dict_path, "r") as f:
            for line in f.readlines():
                splited_list = line.strip().split("\t")
                pinyin = splited_list[0]
                words_dict[pinyin.replace(" ", "")] = splited_list[1:]
        return words_dict

    
    def get_first_state(self):
        state = kenlm.State()
        self.lm_model.BeginSentenceWrite(state)
        return state


    def update_lm_score(self, state, word):
        cur_state = kenlm.State()
        score = self.lm_model.BaseScore(state, word, cur_state)
        return score, cur_state


    def merge_paths(self):
        paths_merged = {}
        unfinished_path_count = 0
        for path in self.paths:
            if path.is_valid:
                history_str = "".join(path.word_history)
                if history_str not in paths_merged:
                    paths_merged[history_str] = path
                    unfinished_path_count += 1 if len(path.prefix) else 0
                elif path.lm_score > paths_merged[history_str].lm_score:
                    paths_merged[history_str].lm_score = path.lm_score
                    paths_merged[history_str].lm_state = path.lm_state
        self.paths = [v for v in paths_merged.values()]
        self.paths = sorted(self.paths, key=lambda x: x.lm_score, reverse=True)

        # prune here
        self.paths = self.paths[:unfinished_path_count+self.prune_size]


    def add_new_word(self, pinyin):
        new_paths = []
        for path in self.paths:
            if "".join(path.prefix) + pinyin in self.words_dict:  # if match one word, add a new path.
                for word in self.words_dict["".join(path.prefix) + pinyin]:
                    new_prefix = []
                    new_pinyin_history = path.pinyin_history + path.prefix + [pinyin]
                    new_word_history = path.word_history + [word]
                    base_score, new_lm_state = self.update_lm_score(path.lm_state, word)
                    new_lm_score = path.lm_score + base_score

                    new_path = Path(new_prefix, new_pinyin_history, new_word_history, new_lm_score, new_lm_state)
                    new_paths.append(new_path)
            
            path.prefix += [pinyin]
            if len(path.prefix) >= self.max_match_word_len:
                path.is_valid = False
            
        self.paths += new_paths
        self.merge_paths()


    def get_best_result(self):
        self.merge_paths()
        for path in self.paths:
            if len(path.prefix) == 0:
                return path.word_history
        return ""


    def translate_whole_sentence(self, sentence):
        self.reset()
        for pinyin in sentence.split():
            self.add_new_word(pinyin)

        text = self.get_best_result()
        self.reset()
        return text



def test01():
    sentence = "xie2 shou3 kai1 chuang4 ya4 zhou1 an1 quan2 he2 fa1 zhan3 xin1 ju2 mian4"
    
    arpa_file = "/ai-cephfs/Share/hyc_share/lm/trigram.binary"
    dict_path = "/ai-cephfs/Share/hyc_share/lm/pinyin2words.txt"
    lm_tree = LMTree(arpa_file, dict_path)
    
    for pinyin in sentence.split():
        lm_tree.add_new_word(pinyin)

    text = lm_tree.get_best_result()

    print(text)
    

if __name__ == "__main__":
    test01()
