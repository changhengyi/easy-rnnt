import copy
import numpy as np
import torch
import random

def tensor2np(x):
    if x is None:
        return x
    return x.cpu().detach().numpy()


def tensor2scalar(x):
    if isinstance(x, float):
        return x
    return x.cpu().detach().item()


def np2tensor(array, device=None):
    tensor = torch.from_numpy(array).to(device)
    return tensor


def pad_list(xs, pad_value=0., pad_left=False):
    """Convert list of Tensors to a single Tensor with padding.

    Args:
        xs (list): A list of length `[B]`, which contains Tensors of size `[T, input_size]`
        pad_value (float):
        pad_left (bool):
    Returns:
        xs_pad (FloatTensor): `[B, T, input_size]`

    """
    bs = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = xs[0].new_zeros(bs, max_time, * xs[0].size()[1:]).fill_(pad_value)
    for b in range(bs):
        if len(xs[b]) == 0:
            continue
        if pad_left:
            xs_pad[b, -xs[b].size(0):] = xs[b]
        else:
            xs_pad[b, :xs[b].size(0)] = xs[b]
    return xs_pad


def set_batch_size(batch_size, dynamic_batching, num_replicas, df, offset):
    if not dynamic_batching:
        return batch_size
    
    min_xlen = df[offset:offset + 1]['xlen'].values[0]
    min_ylen = df[offset:offset + 1]['ylen'].values[0]
    

    if min_xlen <= 800:
        pass
    elif min_xlen <= 1600 or 80 < min_ylen <= 100:
        batch_size //= 2
    else:
        batch_size //= 8

    batch_size = batch_size // num_replicas * num_replicas
    batch_size = max(num_replicas, batch_size)
    # NOTE: ensure batch size>=1 for all replicas
    return batch_size


def sort_bucketing(df, batch_size, dynamic_batching,
                   num_replicas=1):
    """Bucket utterances in a sorted dataframe. This is also used for evaluation.

    Args:
        batch_size (int): size of mini-batch
        batch_size_type (str): type of batch size counting
        dynamic_batching (bool): change batch size dynamically in training
        num_replicas (int): number of replicas for distributed training
    Returns:
        indices_buckets (List[List]): bucketted utterances

    """
    indices_buckets = []  # list of list
    offset = 0
    indices_rest = list(df.index)
    while True:
        _batch_size = set_batch_size(batch_size, dynamic_batching, num_replicas, df, offset)

        indices = list(df[offset:offset + _batch_size].index)
        if len(indices) >= num_replicas:
            indices_buckets.append(indices)
        offset += len(indices)
        if offset >= len(df):
            break

    return indices_buckets


def shuffle_bucketing(df, batch_size, dynamic_batching,
                      seed=None, num_replicas=1):
    """Bucket utterances having a similar length and shuffle them for Transformer training.

    Args:
        batch_size (int): size of mini-batch
        batch_size_type (str): type of batch size counting
        dynamic_batching (bool): change batch size dynamically in training
        seed (int): seed for randomization
        num_replicas (int): number of replicas for distributed training
    Returns:
        indices_buckets (List[List]): bucketted utterances

    """
    indices_buckets = []  # list of list
    offset = 0
    while True:
        _batch_size = set_batch_size(batch_size, dynamic_batching, num_replicas, df, offset)

        indices = list(df[offset:offset + _batch_size].index)
        if len(indices) >= num_replicas:
            indices_buckets.append(indices)
        offset += len(indices)
        if offset >= len(df):
            break

    # shuffle buckets globally
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices_buckets)
    return indices_buckets


def tensor2np(x):
    """Convert torch.Tensor to np.ndarray.

    Args:
        x (torch.Tensor):
    Returns:
        np.ndarray

    """
    if x is None:
        return x
    return x.cpu().detach().numpy()


def np2tensor(array, device=None):
    """Convert form np.ndarray to torch.Tensor.

    Args:
        array (np.ndarray): A tensor of any sizes
    Returns:
        tensor (torch.Tensor):

    """
    tensor = torch.from_numpy(array).to(device)
    return tensor


def pad_list(xs, pad_value=0., pad_left=False):
    """Convert list of Tensors to a single Tensor with padding.

    Args:
        xs (list): A list of length `[B]`, which contains Tensors of size `[T, input_size]`
        pad_value (float):
        pad_left (bool):
    Returns:
        xs_pad (FloatTensor): `[B, T, input_size]`

    """
    bs = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = xs[0].new_zeros(bs, max_time, * xs[0].size()[1:]).fill_(pad_value)
    for b in range(bs):
        if len(xs[b]) == 0:
            continue
        if pad_left:
            xs_pad[b, -xs[b].size(0):] = xs[b]
        else:
            xs_pad[b, :xs[b].size(0)] = xs[b]
    return xs_pad


def hyp2text(hyps, id2token):
    text = [id2token[hyp_id] for hyp_id in hyps]
    text = "".join(text)
    text = text.replace("▁", " ").strip(" ")
    return text


def compute_wer(ref, hyp, normalize=False):
    """Compute Word Error Rate.

        [Reference]
            https://martin-thoma.com/word-error-rate-calculation/
    Args:
        ref (list): words in the reference transcript
        hyp (list): words in the predicted transcript
        normalize (bool, optional): if True, divide by the length of ref
    Returns:
        wer (float): Word Error Rate between ref and hyp
        n_sub (int): the number of substitution
        n_ins (int): the number of insertion
        n_del (int): the number of deletion

    """
    # Initialisation
    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint16)
    d = d.reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # Computation
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub_tmp = d[i - 1][j - 1] + 1
                ins_tmp = d[i][j - 1] + 1
                del_tmp = d[i - 1][j] + 1
                d[i][j] = min(sub_tmp, ins_tmp, del_tmp)

    wer = d[len(ref)][len(hyp)]

    # Find out the manipulation steps
    x = len(ref)
    y = len(hyp)
    error_list = []
    while True:
        if x == 0 and y == 0:
            break
        else:
            if x > 0 and y > 0:
                if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                    error_list.append("C")
                    x = x - 1
                    y = y - 1
                elif d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                elif d[x][y] == d[x - 1][y - 1] + 1:
                    error_list.append("S")
                    x = x - 1
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif x == 0 and y > 0:
                if d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif y == 0 and x > 0:
                error_list.append("D")
                x = x - 1
            else:
                raise ValueError

    n_sub = error_list.count("S")
    n_ins = error_list.count("I")
    n_del = error_list.count("D")
    n_cor = error_list.count("C")

    assert wer == (n_sub + n_ins + n_del)
    assert n_cor == (len(ref) - n_sub - n_del)

    if normalize:
        wer /= len(ref)

    return wer * 100, n_sub * 100, n_ins * 100, n_del * 100