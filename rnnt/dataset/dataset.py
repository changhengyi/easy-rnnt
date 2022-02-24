import kaldiio
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset


class ASRDataset(Dataset):
    def __init__(self, tsv_path, dict_path, wp_model, min_n_frames=-1, max_n_frames=6000, subsample_factor=4):
        super(Dataset, self).__init__()
        # Load dataset tsv file
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


    def __len__(self):
        return len(self.df)


    @property
    def n_frames(self):
        return self.df['xlen'].sum()


    def __getitem__(self, i):
        # inputs
        feat_path = self.df['feat_path'][i]
        xs = kaldiio.load_mat(feat_path)
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