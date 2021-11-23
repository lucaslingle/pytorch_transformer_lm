import torch as tc
from utils import get_dataloaders
from utils import get_mask
import os


class Runner:
    def __init__(self, dataset_map_fn, batch_map_fn, batch_size, max_tokens, model_name, checkpoint_dir, output_dir):
        self.dataset_map_fn = dataset_map_fn
        self.batch_map_fn = batch_map_fn
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.global_step = 0

    def train_epoch(self, model, train_dataloader, optimizer, scheduler, device):
        for batch_idx, (X, Y, L) in enumerate(train_dataloader, 1):
            X, Y, L = X.to(device), Y.to(device), L.to(device)

            if len(X) < self.batch_size:
                continue

            # Forward
            logprobs, _ = model(X)
            mask = get_mask(L, sequence_len=X.shape[-1])
            logprobs_Y = tc.gather(logprobs, dim=-1, index=Y.unsqueeze(dim=-1)).squeeze(dim=-1)
            masked_nll_loss = -(mask * logprobs_Y).sum()
            loss = masked_nll_loss / mask.sum()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.global_step += 1

            if True: #batch_idx % 100 == 0:
                loss = loss.item()
                print("batch: [{}/{}]... loss: {}".format(batch_idx, 25000 // self.batch_size, loss))

        return

    def evaluate_epoch(self, model, dataloader, device, fast_eval=True):
        num_test_tokens = 0
        test_loss, correct = 0, 0
        with tc.no_grad():
            for batch_idx, (X, Y, L) in enumerate(dataloader, 1):
                if fast_eval and batch_idx % 10 != 0:
                    continue
                X, Y, L = X.to(device), Y.to(device), L.to(device)

                logprobs, _ = model(X)
                mask = get_mask(L, sequence_len=X.shape[-1])
                logprobs_Y = tc.gather(logprobs, dim=-1, index=Y.unsqueeze(dim=-1)).squeeze(dim=-1)
                masked_nll_loss = -(mask * logprobs_Y).sum()
                loss = masked_nll_loss

                test_loss += loss.item()
                correct += (logprobs.argmax(-1) == Y).type(tc.float).sum().item()
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

            # shuffle, batch, and preprocess an ephemeral dataset
            train_dataloader, test_dataloader = get_dataloaders(
                dataset_map_fn=self.dataset_map_fn, batch_map_fn=self.batch_map_fn, batch_size=self.batch_size)

            model.train()
            self.train_epoch(model, train_dataloader, optimizer, scheduler, device)

            model.eval()
            test_eval_dict = self.evaluate_epoch(model, test_dataloader, device)
            test_accuracy = test_eval_dict['accuracy'] * 100
            test_loss = test_eval_dict['loss']
            print(f"Test Error: \n Accuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f}\n")

            epoch += 1

            self.save_checkpoint(model, optimizer)

    def generate(self, model, vocab, num_samples=10):
        go_tokens = vocab.stoi['<go>'] * tc.ones((num_samples, 1)).long()
        i_tokens = vocab.stoi['i'] * tc.ones((num_samples, 1)).long()
        saw_tokens = vocab.stoi['saw'] * tc.ones((num_samples, 1)).long()
        this_tokens = vocab.stoi['this'] * tc.ones((num_samples, 1)).long()
        a_tokens = vocab.stoi['a'] * tc.ones((num_samples, 1)).long()
        few_tokens = vocab.stoi['few'] * tc.ones((num_samples, 1)).long()
        tokens = tc.cat((go_tokens, i_tokens, saw_tokens, this_tokens, a_tokens, few_tokens), dim=-1)

        model.eval()

        with tc.no_grad():
            past = None
            x_tm1 = tokens

            for t in range(tokens.shape[1], self.max_tokens+2):
                print("t: {}".format(t))
                # generate tokens x_t, ..., x_{max_tokens}, x_{max_tokens+1}.
                # in a trained model, ideally the last token would be a '<pad>'
                logprobs, past = model.forward(x_tm1, past=past)
                probs = tc.nn.Softmax(dim=-1)(logprobs[:,-1,:])
                x_t = tc.multinomial(probs, num_samples=1)
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
        return

    def save_checkpoint(self, model, optimizer):
        model_path = os.path.join(self.checkpoint_dir, self.model_name)
        os.makedirs(model_path, exist_ok=True)

        tc.save(model.state_dict(), os.path.join(self.checkpoint_dir, self.model_name, 'model.pth'))
        tc.save(optimizer.state_dict(), os.path.join(self.checkpoint_dir, self.model_name, 'optimizer.pth'))

    def maybe_load_checkpoint(self, model, optimizer):
        try:
            model.load_state_dict(tc.load(os.path.join(self.checkpoint_dir, self.model_name, 'model.pth')))
            optimizer.load_state_dict(tc.load(os.path.join(self.checkpoint_dir, self.model_name, 'optimizer.pth')))
            print('Successfully loaded checkpoint.')
        except Exception:
            print('Bad checkpoint or none. Continuing training from scratch.')