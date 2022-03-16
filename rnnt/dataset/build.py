
from rnnt.dataset.dataset import ASRDataset
from rnnt.dataset.dataloader import ASRDataLoader
from rnnt.dataset.sampler import ASRBatchSampler


def build_dataloader(args, tsv_path, batch_size, cmvn_path=None, num_workers=0, pin_memory=False, distributed=False, resume_epoch=0):

    dataset = ASRDataset(tsv_path=tsv_path,
                            cmvn_path = cmvn_path,
                            dict_path=args.dict,
                            wp_model=args.wp_model,
                            min_n_frames=args.min_n_frames,
                            max_n_frames=args.max_n_frames,
                            subsample_factor=args.subsample_factor)

    batch_sampler = ASRBatchSampler(dataset=dataset,
                                       distributed=distributed,
                                       batch_size=batch_size,
                                       dynamic_batching=args.dynamic_batching,
                                       shuffle_bucket=args.shuffle_bucket,
                                       resume_epoch=resume_epoch)

    dataloader = ASRDataLoader(dataset=dataset,
                                  batch_sampler=batch_sampler,
                                  sort_stop_epoch=args.sort_stop_epoch,
                                  collate_fn=custom_collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory and distributed)

    return dataloader


def custom_collate_fn(data):
    tmp = {k: [data_i[k] for data_i in data] for k in data[0].keys()}

    return tmp