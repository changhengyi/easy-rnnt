from pypinyin import Style, pinyin, lazy_pinyin
from tqdm import tqdm

tone_set = set(["1", "2", "3", "4", "5"])
def words2pinyin(words):
    heteronym = pinyin(words, style=Style.TONE3, heteronym=True)
    result = [[]]
    for word_pinyin_list in heteronym:
        new_result = []
        for res in result:
            for word_pinyin in word_pinyin_list:
                if word_pinyin[-1] not in tone_set:
                    word_pinyin += "5"
                new_result.append(res + [word_pinyin])
        result = new_result
    
    result = [" ".join(res) for res in result]

    lazy_res = lazy_pinyin(words, style=Style.TONE3, neutral_tone_with_five=True)

    res_set = set(result)
    res_set.add(" ".join(lazy_res))
    
    return list(res_set)


def lazy_words2pinyin(words):
    """[]"""
    if len(words) == 1:
        return [py[0] for py in pinyin(words, style=Style.TONE3, heteronym=True)]
    else:
        return [" ".join(lazy_pinyin(words, style=Style.TONE3, neutral_tone_with_five=True))]


def make_pinyin2words_dict():
    words_file = "/ai-cephfs/Share/hyc_share/lm/words_24w.txt"
    words2pinyin_dict, pinyin2words_dict = {}, {}
    with open(words_file, "r") as fr:
        for line in tqdm(fr.readlines()):
            words = line.strip()
            pinyin_list = lazy_words2pinyin(words)
            # print(pinyin_list)
            words2pinyin_dict[words] = pinyin_list
            for pinyin in pinyin_list:
                if pinyin not in pinyin2words_dict:
                    pinyin2words_dict[pinyin] = [words]
                else:
                    pinyin2words_dict[pinyin].append(words)
                # print(pinyin, pinyin2words_dict[pinyin])

    dict_file = "/ai-cephfs/Share/hyc_share/lm/pinyin2words.txt"
    pinyin2words_list = [[k] + pinyin2words_dict[k] for k in pinyin2words_dict]
    pinyin2words_list = sorted(pinyin2words_list, key=lambda x: x)
    with open(dict_file, "w") as fw:
        for line in tqdm(pinyin2words_list):
            if line[0][-1] not in tone_set:
                line[0] = line[0] + "5"
            fw.write("{}\n".format("\t".join(line)))
    

def test01():
    for words in ["朝阳", "长阳", "衣裳", "重"]:
        res = words2pinyin(words)
        print("{}\t{}\n".format(words, "\t".join(res)))


if __name__ == "__main__":
    make_pinyin2words_dict()
