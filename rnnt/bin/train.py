import sys
sys.path.insert(0,"/home/changhengyi/easy-rnnt")
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import logging


from rnnt.dataset.build import build_dataloader
from rnnt.model.asr_model import ASR
from rnnt.bin.args import get_conf
from rnnt.bin.scheduler import Scheduler


def train(args):
    ### init logger
    log_path = args.save_path + "/train.log"
    logging.basicConfig(filename = log_path, level = logging.INFO, format = '%(asctime)s[%(levelname)s]: %(message)s')
    logger = logging.getLogger()

    ### init train loader
    train_loader = build_dataloader(args, args.trainset_path, args.batch_size, num_workers=args.num_workers, pin_memory=False, distributed=True)
    test_loader = build_dataloader(args, args.testset_path, 1, num_workers=args.num_workers)

    ### init model
    model = ASR(args)
    n = torch.cuda.device_count() // args.local_world_size
    device_ids = list(range(args.local_rank * n, (args.local_rank + 1) * n))
    torch.cuda.set_device(device_ids[0])
    model.cuda(device_ids[0])
    model = DDP(model, device_ids=device_ids)
    num_replicas = args.local_world_size

    ### init optimizer and scheduler
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = Scheduler(args, optimizer, logger)

    if args.local_rank == 0:
        scheduler.eval_wordpiece(model.module, test_loader, args.local_rank)

    return

    ### start training
    for epoch in range(args.n_epochs):
        if args.local_rank == 0:
            pbar_epoch = tqdm(total=len(train_loader))
            
        
        for batch_data in train_loader:
            num_samples = len(batch_data['utt_ids']) * num_replicas

            loss, observation = model(batch_data)
            loss *= num_replicas
            loss.backward()
            loss.detach()
            del loss
            scheduler.add_observation(observation)
            scheduler.step()
            scheduler.zero_grad(set_to_none=True)

            if args.local_rank == 0:
                pbar_epoch.update(num_samples)

        if args.local_rank == 0:
            scheduler.eval_wordpiece(model.module, test_loader, args.local_rank)
            scheduler.save(model.module)
            pbar_epoch.close()
        
        train_loader.reset(is_new_epoch=True)
        


def main(args):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend=args.dist_backend)
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    args.local_rank = dist.get_rank()
    train(args)

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_conf(OmegaConf.from_cli())
    main(args)