import os

import torch as tc

from utils import get_dataloaders
from utils import get_mask


class Runner:
    def __init__(
            self,
            dataset_map_fn,
            batch_map_fn,
            batch_size,
            max_tokens,
            model_name,
            checkpoint_dir,
            output_dir
    ):
        """
        :param dataset_map_fn: function to apply to dataset rows
        :param batch_map_fn: function to apply to dataset batches
        :param batch_size: number of sequences per batch
        :param max_tokens: number of natural language tokens per sequence,
                           excluding go tokens and padding.
        :param model_name: model name for checkpointing.
        :param checkpoint_dir: checkpoint directory for checkpointing.
        :param output_dir: output directory for saving samples from the model.
        """
        self.dataset_map_fn = dataset_map_fn
        self.batch_map_fn = batch_map_fn
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.global_step = 0

    def _train_epoch(self, model, dataloader, optimizer, scheduler, device):
        model.train()
        for batch_idx, (X, Y, L) in enumerate(dataloader, 1):
            X, Y, L = X.to(device), Y.to(device), L.to(device)

            if len(X) < self.batch_size:
                continue

            # Forward
            logits, _ = model(X)
            mask = get_mask(L, sequence_len=X.shape[-1])
            logprobs_Y = tc.distributions.Categorical(logits=logits).log_prob(Y)
            masked_nll_loss = -(mask * logprobs_Y).sum()
            loss = masked_nll_loss / mask.sum()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.global_step += 1

            # Logging
            loss = loss.item()
            n = 25000 // self.batch_size
            print(f"batch: [{batch_idx}/{n}]... loss: {loss}")

    @tc.no_grad()
    def _evaluate_epoch(self, model, dataloader, device, fast_eval=True):
        model.eval()
        num_test_tokens = 0
        test_loss, correct = 0, 0
        for batch_idx, (X, Y, L) in enumerate(dataloader, 1):
            if fast_eval and batch_idx % 10 != 0:
                continue
            X, Y, L = X.to(device), Y.to(device), L.to(device)

            logits, _ = model(X)
            mask = get_mask(L, sequence_len=X.shape[-1])
            logprobs_Y = tc.distributions.Categorical(logits=logits).log_prob(Y)
            masked_nll_loss = -(mask * logprobs_Y).sum()
            loss = masked_nll_loss

            test_loss += loss.item()
            correct += tc.eq(logits.argmax(-1), Y).float().sum().item()
            num_test_tokens += mask.sum()

        test_loss /= num_test_tokens
        correct /= num_test_tokens
        return {
            "accuracy": correct,
            "loss": test_loss
        }

    def train(self, max_steps, model, optimizer, scheduler, device):
        epoch = 1
        while self.global_step < max_steps:
            print(f"Epoch {epoch}\n-------------------------------")

            # dataloader for this dataset is ephemeral (it's an iterator)
            train_dataloader, test_dataloader = get_dataloaders(
                dataset_map_fn=self.dataset_map_fn,
                batch_map_fn=self.batch_map_fn,
                batch_size=self.batch_size)

            self._train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device)

            test_eval_dict = self._evaluate_epoch(
                model=model,
                dataloader=test_dataloader,
                device=device)

            test_acc = test_eval_dict['accuracy'] * 100
            test_loss = test_eval_dict['loss']
            print(f"Test Error: ")
            print(f"Accuracy: {test_acc:>0.1f}%, Avg loss: {test_loss:>8f}")
            print("")

            self.save_checkpoint(model, optimizer)
            epoch += 1

    @tc.no_grad()
    def generate(self, model, vocab, num_samples=10):
        prefix = ['<go>'] + "i saw this a few".split(" ")
        prefix = [vocab.stoi(token) for token in prefix]
        prefix = tc.tile(tc.LongTensor(prefix).view(1, -1), [num_samples, 1])
        tokens = prefix

        past = None
        x_tm1 = tokens

        model.eval()
        for t in range(tokens.shape[1], self.max_tokens+2):
            print("t: {}".format(t))
            # generate tokens x_t, ..., x_{max_tokens}, x_{max_tokens+1}.
            # in a trained model, ideally the last token would be a '<pad>'
            logits, past = model.forward(x_tm1, past=past)
            x_t = tc.distributions.Categorical(logits=logits).sample()
            tokens = tc.cat((tokens, x_t), dim=-1)
            x_tm1 = x_t

        lines = [' '.join([vocab.itos[x] for x in line]) for line in tokens.numpy()]

        sample_dir = os.path.join(self.output_dir, self.model_name)
        os.makedirs(sample_dir, exist_ok=True)
        fp = os.path.join(sample_dir, 'samples.txt')
        with open(fp, 'a+') as f:
            for line in lines:
                f.write('-' * 80 + '\n')
                f.write(line + '\n')
        print('Generated samples were successfully written to {}'.format(fp))

    def save_checkpoint(self, model, optimizer):
        model_dir = os.path.join(self.checkpoint_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        model_fp = os.path.join(model_dir, 'model.pth')
        optimizer_fp = os.path.join(model_dir, 'optimizer.pth')

        tc.save(model.state_dict(), model_fp)
        tc.save(optimizer.state_dict(), optimizer_fp)

    def maybe_load_checkpoint(self, model, optimizer):
        try:
            model_dir = os.path.join(self.checkpoint_dir, self.model_name)
            model_fp = os.path.join(model_dir, 'model.pth')
            optimizer_fp = os.path.join(model_dir, 'optimizer.pth')

            model.load_state_dict(tc.load(model_fp))
            optimizer.load_state_dict(tc.load(optimizer_fp))
            print('Successfully loaded checkpoint.')
        except Exception:
            print('Bad checkpoint or none. Continuing training from scratch.')
