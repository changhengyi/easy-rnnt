import sys
from multiprocessing import Pool

from tqdm import tqdm


class WordPath:
    def __init__(self, prefix, word_history):
        self.prefix = prefix
        self.word_history = word_history
        self.is_valid = True


class WordsSegment:
    def __init__(self):
        self.max_match_word_len = 7
        self.limit_word_set = set()
        with open("/ai-cephfs/Share/hyc_share/lm/words_24w.txt", "r") as f:
            for line in f.readlines():
                self.limit_word_set.add(line.strip())
        self.paths = [WordPath([], [])]


    def reset(self):
        self.paths = [WordPath([], [])]
    
    
    def merge_paths(self):
        paths_merged = {}
        for path in self.paths:
            if path.is_valid:
                history_str = "".join(path.word_history)
                if history_str not in paths_merged:
                    paths_merged[history_str] = path
                elif len(path.word_history) < len(paths_merged[history_str].word_history):
                    paths_merged[history_str].word_history = path.word_history
        self.paths = [v for v in paths_merged.values()]


    def add_new_word(self, word):
        new_paths = []
        for path in self.paths:
            if "".join(path.prefix) + word in self.limit_word_set:  # if match one word, add a new path.
                new_prefix = []
                new_word_history = path.word_history + ["".join(path.prefix) + word]
                new_path = WordPath(new_prefix, new_word_history)
                new_paths.append(new_path)
            
            path.prefix += [word]
            if len(path.prefix) >= self.max_match_word_len:
                path.is_valid = False
            
        self.paths += new_paths
        self.merge_paths()

    
    def get_best_result(self):
        self.paths = [path for path in self.paths if len(path.prefix) == 0]
        return sorted(self.paths, key=lambda x: len(x.word_history))[0].word_history

    
    def cut(self, sentence):
        self.reset()
        sentence = sentence.replace(" ", "")
        for word in sentence:
            self.add_new_word(word)
        return self.get_best_result()



def error_callback_fn(x):
    print(x)
    import traceback
    traceback.print_exc()


def do_request_block(thread_id):
    word_seg = WordsSegment()
    raw_file = "/ai-cephfs/Share/hyc_share/lm/data/raw.txt{0:02d}".format(thread_id)
    out_file = "/ai-cephfs/Share/hyc_share/lm/data/split.txt{0:02d}".format(thread_id)
    with open(raw_file, "r") as fr, open(out_file, "w") as fw:
        for line in tqdm(fr.readlines()):
            line = line.strip().replace(" ", "")
            words_list = word_seg.cut(line)
            for word in words_list:
                if word not in word_seg.limit_word_set:
                    print(word)
                    print(words_list)
                    return
            fw.write("{}\n".format(" ".join(words_list)))



if __name__ == "__main__":
    num_procs = 40

    pool = Pool(num_procs)

    for i in range(num_procs):
        pool.apply_async(do_request_block, (i,), error_callback=error_callback_fn)

    pool.close()
    pool.join()
