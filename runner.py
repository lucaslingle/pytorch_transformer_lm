import torch as tc
from utils import get_dataloaders
from utils import get_mask


class Runner:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def train_epoch(self, model, train_dataloader, optimizer, device):
        for batch_idx, (X, Y, L) in enumerate(train_dataloader, 1):
            X, Y, L = X.to(device), Y.to(device), L.to(device)

            # Forward
            logprobs, _ = model(X, L)
            mask = get_mask(L, sequence_len=X.shape[-1])
            logprobs_Y = tc.gather(logprobs, dim=-1, index=Y.unsqueeze(dim=-1)).squeeze(dim=-1)
            masked_nll_loss = -(mask * logprobs_Y).sum()
            loss = masked_nll_loss / mask.sum()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose and batch_idx % 100 == 0:
                loss = loss.item()
                print("batch: {}... loss: {}".format(batch_idx, loss))

        return

    def evaluate_epoch(self, model, dataloader, device):
        num_test_tokens = 0
        test_loss, correct = 0, 0
        with tc.no_grad():
            for X, Y, L in dataloader:
                X, Y, L = X.to(device), Y.to(device), L.to(device)

                logprobs, _ = model(X, L)
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

    def train(self, dataset_map_fn, batch_size, epochs, model, device, optimizer):
        for epoch in range(1, epochs+1):
            if self.verbose:
                print(f"Epoch {epoch}\n-------------------------------")

            train_dataloader, test_dataloader = get_dataloaders(dataset_map_fn=dataset_map_fn, batch_size=batch_size)

            model.train()
            self.train_epoch(model, train_dataloader, optimizer, device)

            model.eval()
            test_eval_dict = self.evaluate_epoch(model, test_dataloader, device)
            test_accuracy = test_eval_dict['accuracy'] * 100
            test_loss = test_eval_dict['loss']
            if self.verbose:
                print(f"Test Error: \n Accuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f}\n")

            if epoch % 10 == 0:
                tc.save(model.state_dict(), "model.pth")
                tc.save(optimizer.state_dict(), "optimizer.pth")

    def generate(self, vocab, batch_size, model, fp, max_tokens=20):
        go_tokens = vocab.stoi['<go>'] * tc.ones((batch_size, 1)).long()
        tokens = go_tokens

        model.eval()

        with tc.no_grad():
            past = None
            x_tm1 = go_tokens

            for t in range(1, max_tokens+2):
                # generate tokens x_1, ..., x_{max_tokens}, x_{max_tokens+1}.
                # after training model, the last token should be a '<pad>' token, which serves as eos.
                logprobs, present = model.forward(x_tm1, past=past)
                probs = tc.nn.Softmax(dim=-1)(logprobs.squeeze(dim=1))
                x_t = tc.multinomial(probs, num_samples=1)
                tokens = tc.cat((tokens, x_t), dim=-1)
                x_tm1 = x_t
                past = tc.cat((past, present), dim=-2) # [batch, layer, kvstack, timestep, features].

        lines = [' '.join([vocab.itos[x] for x in line]) for line in tokens.numpy()]

        with open(fp, 'a+') as f:
            for line in lines:
                f.write(line + '\n')

        return


