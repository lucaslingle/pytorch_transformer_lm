import torch as tc
import torchtext as tt
from collections import Counter
from functools import partial
from torch.nn.utils.rnn import pad_sequence


def get_tokenizer():
    tokenizer = tt.data.utils.get_tokenizer('basic_english')
    return tokenizer


def get_vocab(tokenizer):
    # https://pytorch.org/text/stable/vocab.html
    # vocab object will have stoi, a defaultdict with ('<unk>', '<pad>', other tokens) mapping to (0, 1, ...).
    # lookups for unknown tokens will default to the '<unk>' key, and thus the token 0 as the value.
    train_iter = tt.datasets.IMDB(root='data', split='train')
    counter = Counter()
    for y, X in train_iter:
        counter.update(tokenizer(X))
    vocab = tt.vocab.Vocab(counter, specials=('<unk>', '<pad>', '<go>'), min_freq=50)
    return vocab


def text_pipeline(text, tokenizer, vocab):
    sequence = [vocab.stoi[token] for token in tokenizer(text)]
    return sequence


def lstm_preprocess_pipeline(sequences, max_tokens=20):
    batch_size = len(sequences)
    sequences = [tc.Tensor(s)[0:max_tokens] for s in sequences]

    padded = pad_sequence(sequences, padding_value=1.0, batch_first=True) # <pad> is token at index 1 of vocab.
    go_tokens = 2 * tc.ones(size=(batch_size, 1)) # <go> is token at index 2.
    padded = tc.cat((go_tokens, padded), dim=1) # prepend <go> tokens.
    pad_tokens = tc.ones(size=(batch_size, 1)) # <pad> is token at index 1.
    padded = tc.cat((padded, pad_tokens), dim=1) # append one extra <pad> token to each seq. to ensure eos targ.
    padded = padded.long()

    input_tokens = padded[:, 0:-1]
    target_tokens = padded[:, 1:]
    lengths = tc.Tensor([len(s)+1 for s in sequences]) # add 1 to ensure num lstm iters includes input length w go token

    return input_tokens, target_tokens, lengths


def collate_batch(batch, dataset_map_fn, batch_map_fn):
    sequences = [dataset_map_fn(y,x) for y,x in batch]
    X, Y, L = batch_map_fn(sequences)
    # ^ note this differs from other projects since we use the entire batch to choose pad length;
    # this is inherently a non-local computation and cannot be done as fixed per-element map.
    return X, Y, L


def get_dataloaders(dataset_map_fn, batch_size):
    training_data = tt.datasets.IMDB(root='data', split='train')
    test_data = tt.datasets.IMDB(root='data', split='test')

    training_data = tc.utils.data.BufferedShuffleDataset(training_data, buffer_size=25000)
    test_data = tc.utils.data.BufferedShuffleDataset(test_data, buffer_size=25000)

    collate_fn = partial(collate_batch, dataset_map_fn=dataset_map_fn, batch_map_fn=lstm_preprocess_pipeline)

    train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, test_dataloader


def get_mask(lengths, sequence_len):
    batch_size = lengths.shape[0]
    bool_mask = tc.less(
        tc.arange(sequence_len).expand(batch_size, sequence_len),
        lengths.unsqueeze(dim=1).expand(batch_size, sequence_len)
    )
    return bool_mask.float()