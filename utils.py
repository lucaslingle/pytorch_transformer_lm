from collections import Counter
from functools import partial

import torch as tc
import torchtext as tt
from torch.nn.utils.rnn import pad_sequence


def get_tokenizer():
    tokenizer = tt.data.utils.get_tokenizer('basic_english')
    return tokenizer


def get_vocab(tokenizer):
    # https://pytorch.org/text/stable/vocab.html
    # vocab object will have stoi attribute, a defaultdict with
    #     ('<unk>', '<pad>', other tokens) mapping to (0, 1, ...).
    # lookups for unknown tokens will default to the '<unk>' key,
    #    and thus the token 0 as the value.

    train_iter = tt.datasets.IMDB(root='data', split='train')
    counter = Counter()
    for y, X in train_iter:
        counter.update(tokenizer(X))
    vocab = tt.vocab.Vocab(
        counter, specials=('<unk>', '<pad>', '<go>'), min_freq=50)
    return vocab


def text_pipeline(text, tokenizer, vocab):
    sequence = [vocab.stoi[token] for token in tokenizer(text)]
    return sequence


def sequence_preprocess_pipeline(sequences, max_tokens=20):
    batch_size = len(sequences)
    sequences = [tc.Tensor(s)[0:max_tokens] for s in sequences]

    # <pad> is token at index 1 of vocab.
    # note: padding_value arg of this function has to be a float
    #  we cast everything to dtype long to fix this later.
    padded = pad_sequence(sequences, padding_value=1.0, batch_first=True)

    # <go> is token at index 2.
    go_tokens = 2 * tc.ones(size=(batch_size, 1))

    # prepend <go> tokens.
    padded = tc.cat((go_tokens, padded), dim=1)

    # <pad> is token at index 1.
    pad_tokens = tc.ones(size=(batch_size, 1))

    # append one extra <pad> token to each seq. to ensure eos targ.
    padded = tc.cat((padded, pad_tokens), dim=1)
    padded = padded.long()

    # prepare input tokens and right-shifted target tokens.
    input_tokens = padded[:, 0:-1]
    target_tokens = padded[:, 1:]

    # record the unpadded lengths of target tokens including one pad (eos) token
    # so that we can mask out the log probs of the additional pad tokens later.
    lengths = tc.Tensor([len(s)+1 for s in sequences])

    return input_tokens, target_tokens, lengths


def collate_batch(batch, dataset_map_fn, batch_map_fn):
    sequences = [dataset_map_fn(y,x) for y,x in batch]
    X, Y, L = batch_map_fn(sequences)
    return X, Y, L


def get_dataloaders(dataset_map_fn, batch_map_fn, batch_size):
    training_data = tt.datasets.IMDB(root='data', split='train')
    test_data = tt.datasets.IMDB(root='data', split='test')

    training_data = tc.utils.data.BufferedShuffleDataset(
        training_data, buffer_size=25000)
    test_data = tc.utils.data.BufferedShuffleDataset(
        test_data, buffer_size=25000)

    collate_fn = partial(
        collate_batch,
        dataset_map_fn=dataset_map_fn,
        batch_map_fn=batch_map_fn)

    train_dataloader = tc.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        collate_fn=collate_fn)
    test_dataloader = tc.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=collate_fn)

    return train_dataloader, test_dataloader


def get_mask(lengths, sequence_len):
    batch_size = lengths.shape[0]
    bool_mask = tc.less(
        tc.arange(sequence_len).expand(batch_size, sequence_len),
        lengths.unsqueeze(dim=1).expand(batch_size, sequence_len)
    )
    return bool_mask.float()


def get_weight_decay_param_groups(
        model, weight_decay, skip_list=('bias', 'beta', 'gamma')
):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif len(param.shape) == 1 or name.split('.')[-1] in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay, 'weight_decay': weight_decay}]
