import torch
import os
from rnnt.utils import hyp2text_char, compute_wer, hyp2text_wp
from tqdm import tqdm


class Scheduler(object):
    def __init__(self, args, optimizer, logger):
        self.optimizer = optimizer
        self.args = args
        self.logger = logger

        self._epoch = 1
        self._step = 1

        self.lr = args.lr

        # load dict to make id2token
        self.id2token = {0: "<blank>"}
        with open(args.dict, "r") as f:
            for line in f.readlines():
                token, idx = line.strip().split(" ")
                idx = int(eval(idx))
                self.id2token[idx] = token


        if args.resume != "":
            self.optimizer.load(torch.load(args.resume,map_location=self.device)["optimizer_state_dict"])
    
    
    def _update_lr(self):
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            if isinstance(self.optimizer, torch.optim.Adadelta):
                param_group['eps'] = self.lr
            else:
                param_group['lr'] = self.lr
        self.logger.info(f"Learning rate update to {self.lr}")

    
    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)


    def step(self):
        self.optimizer.step()
        self._step += 1


    def epoch(self):
        self._step = 0
        self._epoch += 1
        if self._epoch > self.args.lr_decay_start_epoch:
            self.lr *= self.args.lr_decay_rate
        self._update_lr()


    def eval_testset(self, model, dataloader_list, rank, mode='char'):
        for dataloader in dataloader_list:
            print("Testset: {} Length: {}".format(dataloader.name, len(dataloader)))
            if rank == 0:
                pbar = tqdm(total=len(dataloader))
            wer, n_sub_w, n_ins_w, n_del_w, n_word = 0, 0, 0, 0, 0
            for batch in dataloader:
                num_samples = len(batch['utt_ids'])
                hyps = model.decode_greedy(batch['xs'])
                assert len(hyps) == len(batch['text'])
                for i in range(len(hyps)):
                    ref = batch['text'][i]
                    if mode == 'char':
                        hyp = hyp2text_char(hyps[i], self.id2token)
                    elif mode == 'wp':
                        hyp = hyp2text_wp(hyps[i], self.id2token)

                    self.logger.debug("{} | ref: {}".format(batch['utt_ids'][i], ref))
                    self.logger.debug("{} | hyp: {}".format(batch['utt_ids'][i], hyp))
                    
                    err_b, sub_b, ins_b, del_b = compute_wer(ref.split(" "), hyp.split(" "))
                    wer += err_b
                    n_sub_w += sub_b
                    n_ins_w += ins_b
                    n_del_w += del_b
                    n_word += len(ref.split(' '))

                if rank == 0:
                    pbar.update(num_samples)

            wer /= n_word
            n_sub_w /= n_word
            n_ins_w /= n_word
            n_del_w /= n_word

            self.logger.info('Dataset: {}\t| WER: {:.2f} %% SUB: {:.2f} / INS: {:.2f} / DEL: {:.2f}'.format(dataloader.name, wer, n_sub_w, n_ins_w, n_del_w))
            dataloader.reset(is_new_epoch=True)
            if rank == 0:
                pbar.close()

    
    def add_observation(self, loss_dict):
        self.logger.info("Step: {} | Train Loss: {:.3f} | Dev Loss: {:.3f}".format(self._step, loss_dict['train_loss'], loss_dict["dev_loss"]))


    def save(self, model, remove_old=True):
        save_path = "{}/model.epoch-{}".format(self.args.save_path, self._epoch)
        if remove_old:
            for root, _, names in os.walk(save_path):
                for name in names:
                    if name.startswith("model.epoch-"):
                        os.remove(root + "/" + name)
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}, save_path)

        self.logger.info(f"Model saved at {save_path}")