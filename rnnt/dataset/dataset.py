import kaldiio
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset


class ASRDataset(Dataset):
    def __init__(self, tsv_path, cmvn_path=None, dict_path=None, wp_model=None, min_n_frames=-1, max_n_frames=6000, subsample_factor=4):
        super(Dataset, self).__init__()
        # Load dataset tsv file
        self.name = tsv_path.split("/")[-1].rsplit(".", 1)[0]
        chunk = pd.read_csv(tsv_path, encoding='utf-8',
                            delimiter='\t', chunksize=1000000)
        df = pd.concat(chunk)
        df = df.loc[:, ['utt_id', 'speaker', 'feat_path',
                        'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]

        # remove some invalid uttrances
        n_utts = len(df)
        df = df[df.apply(lambda x: min_n_frames <= x['xlen'] <= max_n_frames, axis=1)]
        df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
        print(f"Removed {n_utts - len(df)} utterances (threshold)")
        n_utts = len(df)
        df = df[df.apply(lambda x: x['ylen'] <= (x['xlen'] // subsample_factor), axis=1)]
        print(f"Removed {n_utts - len(df)} utterances (for CTC)")

        # sort by uttrance length
        df = df.sort_values(by=['xlen'], ascending=True)
        self.df = df.reset_index()

        if cmvn_path is not None:
            print(f"Load cmvn: {cmvn_path}")
            # self.cmvn = self.load_cmvn(cmvn_path)
            self.cmvn = kaldiio.load_mat(cmvn_path)
        else:
            self.cmvn = None


    def __len__(self):
        return len(self.df)


    @property
    def n_frames(self):
        return self.df['xlen'].sum()


    def __getitem__(self, i):
        # inputs
        feat_path = self.df['feat_path'][i]
        xs = kaldiio.load_mat(feat_path)
        if self.cmvn is not None:
            xs = self.apply_cmvn(xs)
        xlen = self.df['xlen'][i]

        # main outputs
        text = self.df['text'][i]
        ys = list(map(int, str(self.df['token_id'][i]).split()))

        mini_batch_dict = {
            'xs': xs,
            'xlens': xlen,
            'ys': ys,
            'utt_ids': self.df['utt_id'][i],
            'speakers': self.df['speaker'][i],
            'text': text
        }

        return mini_batch_dict


    def load_cmvn(self, cmvn_path):
        with open(cmvn_path, "r") as f:
            cmvn1 = [eval(x) for x in f.readline().strip().split()]
            cmvn2 = [eval(x) for x in f.readline().strip().split()]
        cmvn = [cmvn1, cmvn2]
        cmvn = np.array(cmvn)
        return cmvn


    def apply_cmvn(self, feat):    
        stats_0, stats_1 = self.cmvn[0], self.cmvn[1]
        dim = 80
        fream_num = feat.shape[0]
        count = stats_0[dim]
        norm = np.zeros([2,80])
        for d in range(80):
            mean = stats_0[d]/count
            var = (stats_1[d]/count) - mean*mean
            floor = 1.0e-20
            if var < floor:
                print("Flooring cepstral variance from {} to {}".format(var, floor))
                var = floor
            scale = 1.0 / np.sqrt(var)
            if scale != scale or 1/scale == 0.0:
                print("ERROR:NaN or infinity in cepstral mean/variance computation")
                return  
            offset = -(mean*scale)
            norm[0][d] = offset
            norm[1][d] = scale
        res = np.zeros([fream_num, dim])
        for t in range(fream_num):
            for d in range(dim):
                res[t][d] = feat[t][d] * norm[1][d] + norm[0][d]
        return res