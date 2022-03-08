import numpy as np
from torch.utils.data import DataLoader


class ASRDataLoader(DataLoader):
    def __init__(self, dataset, batch_sampler, sort_stop_epoch,
                 num_workers=0, collate_fn=None, pin_memory=False):

        super().__init__(dataset=dataset,
                         shuffle=False,
                         sampler=None,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         drop_last=False)
        # NOTE: dynamic batch size and shuffling are controlled in batch_sampler

        np.random.seed(1)

        self.epoch = 0  # counter
        self.sort_stop_epoch = sort_stop_epoch
        self.name = dataset.name

    def __len__(self):
        """Number of utterances."""
        return len(self.dataset)

    @property
    def epoch_detail(self):
        """Progress of the current epoch."""
        epoch_ratio = self.batch_sampler.offset / len(self)
        # NOTE: this is not accurate when num_workers > 0
        return epoch_ratio

    @property
    def n_frames(self):
        return self.dataset.n_frames

    def reset(self, batch_size=None, is_new_epoch=False):
        if is_new_epoch:
            self.epoch += 1

            # shuffle the whole data per epoch (sort -> shuffle)
            if self.epoch >= self.sort_stop_epoch:
                self.batch_sampler.shuffle_bucket = True

                # This changes not only the order of buckets but also how buckets are constructed
                self.batch_sampler.df = self.batch_sampler.df.reindex(np.random.permutation(self.batch_sampler.df.index))

                # Re-indexing
                self.batch_sampler.df = self.batch_sampler.df.reset_index()

        self.batch_sampler.reset(batch_size, epoch=self.epoch)
