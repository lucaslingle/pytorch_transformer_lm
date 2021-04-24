import argparse
import torch as tc
from utils import get_tokenizer, get_vocab, text_pipeline, sequence_preprocess_pipeline
from functools import partial
from model import PreactivationTranformer
from runner import Runner

# Parse arguments.
parser = argparse.ArgumentParser('Pytorch Transformer Language Model')
parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_context_size', type=int, default=20)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='dir name for all checkpoints generated')
parser.add_argument('--model_name', type=str, default='model', help='model name used for checkpoints and samples')
parser.add_argument('--epochs', type=int, default='10')
args = parser.parse_args()

# Preprocessing.
tokenizer = get_tokenizer()
vocab = get_vocab(tokenizer)
text_preprocessing = partial(text_pipeline, tokenizer=tokenizer, vocab=vocab)
dataset_map_fn = lambda y,x: text_preprocessing(x)
batch_map_fn = partial(sequence_preprocess_pipeline, max_tokens=args.max_context_size)

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

# Runner.
runner = Runner(
    dataset_map_fn=dataset_map_fn,
    batch_map_fn=batch_map_fn,
    batch_size=args.batch_size,
    context_size=args.max_context_size,
    model_name=args.model_name,
    checkpoint_dir=args.checkpoint_dir,
    output_dir=args.output_dir)

runner.maybe_load_checkpoint(model, optimizer)

if args.mode == 'train':
    runner.train(epochs=args.epochs, model=model, optimizer=optimizer, scheduler=None, device=device)
elif args.mode == 'generate':
    runner.generate(model=model, vocab=vocab),
else:
    raise NotImplementedError

