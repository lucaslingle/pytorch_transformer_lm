import argparse

import torch as tc

from utils import (
    get_tokenizer,
    get_vocab,
    text_pipeline,
    sequence_preprocess_pipeline,
    get_weight_decay_param_groups,
)
from model import PreactivationTranformer
from runner import Runner


def create_argparser():
    # Parse arguments.
    parser = argparse.ArgumentParser(
        'Pytorch implementation of a Transformer-based Language Model.')
    parser.add_argument('--mode', choices=['train', 'generate'], default='train')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=200)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--max_lr', type=int, default=2.5e-4)
    parser.add_argument('--weight_decay', type=int, default=0.01)
    parser.add_argument('--max_steps', type=int, default=100 * 195)
    parser.add_argument('--model_name', type=str, default='model',
                        help='model name used for checkpoints and samples')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='dir name for all checkpoints generated')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='dir name for all samples generated')
    return parser


def create_preprocessing(max_tokens):
    tokenizer = get_tokenizer()
    vocab = get_vocab(tokenizer)
    dataset_map_fn = lambda y,x: text_pipeline(
        x, tokenizer=tokenizer, vocab=vocab)
    batch_map_fn = lambda sequences: sequence_preprocess_pipeline(
        sequences, max_tokens=max_tokens)
    return vocab, tokenizer, dataset_map_fn, batch_map_fn


def get_device():
    dev = "cuda" if tc.cuda.is_available() else "cpu"
    return dev


def create_model(vocab, max_tokens, n_layers, d_model, n_heads, device):
    model = PreactivationTranformer(
        n_vocab=len(vocab.stoi),
        n_ctx=max_tokens+1,
        n_emb=d_model,
        n_heads=n_heads,
        n_layers=n_layers).to(device)
    return model


if __name__ == '__main__':
    args = create_argparser().parse_args()

    # get preprocessing ops.
    vocab, tokenizer, dataset_map_fn, batch_map_fn = create_preprocessing(
        args.max_tokens)

    # get device.
    device = get_device()
    print(f'Using {device}.')

    # create learning system.
    model = create_model(
        vocab=vocab,
        max_tokens=args.max_tokens,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        device=device)
    print(model)

    optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(model, args.weight_decay),
        lr=args.max_lr)

    scheduler = tc.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.max_steps,
        eta_min=0.0)

    # create runner.
    runner = Runner(
        dataset_map_fn=dataset_map_fn,
        batch_map_fn=batch_map_fn,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir)

    runner.maybe_load_checkpoint(model, optimizer, scheduler)

    # run it.
    if args.mode == 'train':
        runner.train(
            max_steps=args.max_steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device)

    if args.mode == 'generate':
        runner.generate(model=model, vocab=vocab)
