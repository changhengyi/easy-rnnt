import logging
import numpy as np
import random
import torch.distributed as dist
from rnnt.utils import shuffle_bucketing, sort_bucketing

logger = logging.getLogger(__name__)

if dist.is_available():
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler
else:
    from torch.utils.data.sampler import BatchSampler
    sampler = BatchSampler


class ASRBatchSampler(sampler):
    def __init__(self, dataset, distributed, batch_size, dynamic_batching, shuffle_bucket=False, seed=1, resume_epoch=0):
        """Custom BatchSampler.

        Args:
            dataset (Dataset): pytorch Dataset class
            batch_size (int): size of mini-batch
            dynamic_batching (bool): change batch size dynamically in training
            shuffle_bucket (bool): gather similar length of utterances and shuffle them
            seed (int): seed for randomization
            resume_epoch (int): epoch to resume training
        """
        if distributed:
            super().__init__(dataset=dataset,
                             num_replicas=dist.get_world_size(),
                             rank=dist.get_rank())
        else:
            self.rank = 0
            self.num_replicas = 1
            self.total_size = len(dataset.df.index) * self.num_replicas

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.df = dataset.df
        self.batch_size = batch_size * self.num_replicas
        if self.num_replicas > 1 and self.rank == 0:
            logger.info(f"Batch size is automatically increased from {batch_size} to {self.batch_size}.")
        self.dynamic_batching = dynamic_batching
        self.shuffle_bucket = shuffle_bucket

        self._offset = 0
        # NOTE: epoch should not be counted in BatchSampler

        if shuffle_bucket:
            self.indices_buckets = shuffle_bucketing(self.df, self.batch_size, self.dynamic_batching,
                                                     seed=seed + resume_epoch,
                                                     num_replicas=self.num_replicas)
        else:
            self.indices_buckets = sort_bucketing(self.df, self.batch_size, self.dynamic_batching,
                                                  num_replicas=self.num_replicas)
        self._iteration = len(self.indices_buckets)


    def __len__(self):
        """Number of mini-batches."""
        return self._iteration


    def __iter__(self):
        while True:
            indices, is_new_epoch = self.sample_index()
            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            if is_new_epoch:
                self.reset()
            yield indices
            if is_new_epoch:
                break


    @property
    def offset(self):
        return self._offset


    def reset(self, batch_size=None, epoch=0):
        """Reset data counter and offset.

            Args:
                batch_size (int): size of mini-batch
                epoch (int): current epoch

        """
        if batch_size is None:
            batch_size = self.batch_size

        self._offset = 0

        if self.shuffle_bucket:
            self.indices_buckets = shuffle_bucketing(self.df, batch_size, self.dynamic_batching,
                                                     seed=self.seed + epoch,
                                                     num_replicas=self.num_replicas)
        else:
            self.indices_buckets = sort_bucketing(self.df, batch_size, self.dynamic_batching,
                                                  num_replicas=self.num_replicas)
        self._iteration = len(self.indices_buckets)


    def sample_index(self):
        """Sample data indices of mini-batch.

        Returns:
            indices (np.ndarray): indices of dataframe in the current mini-batch

        """
        indices = self.indices_buckets.pop(0)
        self._offset += len(indices)
        is_new_epoch = (len(self.indices_buckets) == 0)

        if self.shuffle_bucket:
            # Shuffle utterances in a mini-batch
            indices = random.sample(indices, len(indices))

        return indices, is_new_epoch
