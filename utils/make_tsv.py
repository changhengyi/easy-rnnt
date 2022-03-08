"""
you need to prepare following files as kaldi format:

text wav.scp utt2spk spk2utt feats.scp

"""
import kaldiio
from tqdm import tqdm


def make_tsv(text_path, feats_path, dict_path, write_path):
    token2idx = {}
    with open(dict_path, "r") as fdict:
        for line in fdict.readlines():
            token, idx = line.strip().split()
            token2idx[token] = eval(idx)

    data_dict = {}
    with open(text_path, "r") as ftext:
        for line in tqdm(ftext.readlines()):
            utt_id, text = line.strip().split(" ", 1)
            data_dict[utt_id] = [text]
    with open(feats_path, "r") as ffeats:
        for line in tqdm(ffeats.readlines()):
            utt_id, feat = line.strip().split(" ", 1)
            data_dict[utt_id].append(feat)

    

    with open(write_path, "w") as f:
        f.write('utt_id\tspeaker\tfeat_path\txlen\txdim\ttext\ttoken_id\tylen\tydim\n')
        xdim = 80
        ydim = len(token2idx)
        for utt_id in tqdm(data_dict):
            speaker = utt_id
            feat_path = data_dict[utt_id][1]
            xlen = kaldiio.load_mat(feat_path).shape[0]
            text = data_dict[utt_id][0]
            word_list = text.split()
            token_id = " ".join([str(token2idx[x]) for x in word_list])
            ylen = len(word_list)
            f.write(f'{utt_id}\t{speaker}\t{feat_path}\t{xlen}\t{xdim}\t{text}\t{token_id}\t{ylen}\t{ydim}\n')


def main():
    for dataset in ['1', '2', '3', '4']:
        text_path = f"/home/dev-data/speechIO/SPEECHIO_ASR_ZH0000{dataset}/text"
        feats_path = f"/home/dev-data/speechIO/SPEECHIO_ASR_ZH0000{dataset}/feats.scp"
        dict_path = "/home/changhengyi/rnnt-exp/chinese/dict/dict.txt"
        write_path = f"/home/changhengyi/rnnt-exp/chinese/datasets/speechio_{dataset}.tsv"
        make_tsv(text_path, feats_path, dict_path, write_path)

def main2():
    for dataset in ['dev', 'test', 'test_hard', 'train']:
        text_path = f"/home/changhengyi/rnnt-exp/chinese/{dataset}/text"
        feats_path = f"/home/changhengyi/rnnt-exp/chinese/{dataset}/feats.scp"
        dict_path = "/home/changhengyi/rnnt-exp/chinese/dict/dict.txt"
        write_path = f"/home/changhengyi/rnnt-exp/chinese/datasets/{dataset}.tsv"
        make_tsv(text_path, feats_path, dict_path, write_path)

if __name__ == "__main__":
    main2()