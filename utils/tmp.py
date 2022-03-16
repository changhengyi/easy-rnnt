from tqdm import tqdm

def get_word_dict():
    org_file = "/ai-cephfs/Speech/Member/zhangpengshen/wenet2/examples/general-use/data/local/cn-pinyin_en-grapheme/dict/word2pinyin.txt.clean"
    out_file = "/home/changhengyi/easy-rnnt/examples/chinese/pinyin.txt"
    out_file2 = "/home/changhengyi/easy-rnnt/examples/chinese/words.txt"

    pinyin_dict = {}
    with open(org_file, "r") as fr, open(out_file, "w") as fw, open(out_file2, "w") as fw2:
        for line in tqdm(fr.readlines()):
            tmp = line.strip().split(" ")
            words, pinyin_list = tmp[0], tmp[1:]
            if len(words) != len(pinyin_list):
                print(line)
                continue
            assert len(words) == len(pinyin_list)
            for i in range(len(words)):
                pinyin = pinyin_list[i]
                word = words[i]
                if pinyin in pinyin_dict:
                    if word in pinyin_dict[pinyin]:
                        pinyin_dict[pinyin][word] += 1
                    else:
                        pinyin_dict[pinyin][word] = 1
                else:
                    pinyin_dict[pinyin] = {word : 1}

        pinyin_list = [[pinyin, pinyin_dict[pinyin]] for pinyin in pinyin_dict]
        pinyin_list = sorted(pinyin_list, key=lambda x: x[0])

        
        total = 0
        for pinyin, words_dict in tqdm(pinyin_list):
            words_list = [[word, words_dict[word]] for word in words_dict]
            words_list = sorted(words_list, key=lambda x: x[1], reverse=True)
            words_list = [x[0] for x in words_list if x[1] > 0]
            total += len(words_list)
            if len(words_list):
                fw.write("{}\t{}\n".format(pinyin, " ".join(words_list)))
            else:
                fw.write("{}\t{}\n".format(pinyin, "unk"))
        print(total)


def add_original_text():
    for i in range(1,5):
        text_org = "/ai-cephfs/Speech/Member/zhangpengshen/wenet2/examples/test_data/Leaderboard/datasets/SPEECHIO_ASR_ZH0000{}/text.org".format(i)
        testset_path = "/home/changhengyi/rnnt-decode_test/speechio_{}.tsv".format(i)
        write_path = "/home/changhengyi/rnnt-decode_test/speechio_{}.new.tsv".format(i)
        text_dict = {}
        with open(text_org, "r") as f:
            for line in f.readlines():
                utt_id, text = line.strip().split("\t")
                text_dict[utt_id] = text.replace(" ", "")
        with open(testset_path, "r") as fr, open(write_path, "w") as fw:
            fw_lines = fr.readlines()
            fw_lines[0] = fw_lines[0].strip() + "\ttextorg\n"
            for idx in range(1,len(fw_lines)):
                utt_id = fw_lines[idx].strip().split()[0]
                fw_lines[idx] = fw_lines[idx].strip() + "\t{}\n".format(text_dict[utt_id])

            for line in fw_lines:
                fw.write(line)

        
def words_filter():
    single_word_file = "D:/下载/6k_single_words.txt"
    all_words_file = "D:/下载/31w.vocab"

    with open(single_word_file, "r", encoding="utf-8") as fsingle, open(all_words_file, "r", encoding="utf-8") as fall:
        single_word_set = set()
        for line in fsingle.readlines():
            word = line.strip()
            assert len(word) == 1
            single_word_set.add(word)

        final_words_set = set()
        for line in fall.readlines():
            words = line.strip()
            is_valid = True
            for word in words:
                if word not in single_word_set:
                    is_valid = False
            if is_valid:
                final_words_set.add(words)

        for single_word in single_word_set:
            final_words_set.add(single_word)

    res = list(final_words_set)
    res = sorted(res)
    with open("D:/下载/words_24w.txt", "w", encoding="utf-8") as fw:
        for word in res:
            fw.write("{}\n".format(word))

if __name__ == "__main__":
    words_filter()