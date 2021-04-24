import argparse
import torch as tc
from utils import get_tokenizer, get_vocab, text_pipeline
from functools import partial
from model import PreactivationTranformer
from runner import Runner

# Parse arguments.
parser = argparse.ArgumentParser('Pytorch LSTM Language Model')
parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train')
# n_vocab, n_ctx, n_emb, n_heads, n_layers
parser.add_argument('--max_context_size', type=int, default='20')
parser.add_argument('--d_model', type=int, default='128')
parser.add_argument('--n_heads', type=int, default='4')
parser.add_argument('--n_layers', type=int, default='2')
args = parser.parse_args()

# Preprocessing.
tokenizer = get_tokenizer()
vocab = get_vocab(tokenizer)
text_preprocessing = partial(text_pipeline, tokenizer=tokenizer, vocab=vocab, n_tokens=args.max_context_size)
dataset_map_fn = lambda y,x: text_preprocessing(x)
batch_size = 20

# Device.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Model.
model = PreactivationTranformer(
    n_vocab=len(vocab.stoi),
    n_ctx=args.max_context_size,
    n_emb=args.d_model,
    n_heads=args.n_heads,
    n_layers=args.n_layers).to(device)

optimizer = tc.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001) # change this to cosine schedule in a bit

try:
    model.load_state_dict(tc.load("model.pth"))
    optimizer.load_state_dict(tc.load("optimizer.pth"))
    print('successfully reloaded checkpoint.')
except Exception:
    print('no checkpoint found.')

# Runner.
runner = Runner(verbose=True)
epochs = 10

if args.mode == 'train':
    runner.train(dataset_map_fn, batch_size, epochs, model, device, optimizer)
elif args.mode == 'generate':
    runner.generate(vocab, batch_size, model, 'samples.txt')
else:
    raise NotImplementedError

