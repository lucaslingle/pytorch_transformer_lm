import math

import torch as tc


# ported from gpt-2
def split_states(x, n):
    *start, m = x.shape
    shape = start + [n, m//n]
    return x.reshape(*shape)


def merge_states(x):
    *start, a, b = x.shape
    shape = start + [a*b]
    return x.reshape(*shape)


def attention_mask(nd, ns):
    i = tc.arange(nd).view(nd, 1)
    j = tc.arange(ns).view(1, ns)
    m = i >= j - (ns - nd)
    return m.int()


class MultiheadAttention(tc.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.c_attn = tc.nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = tc.nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def split_heads(self, x):
        # [B, T, D] -> [B, H, T, D//H].
        return split_states(x, self.num_heads).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        # [B, H, T, D//H] -> [B, T, D].
        return merge_states(x.permute(0, 2, 1, 3))

    def mask_attn_weights(self, w):
        _, _, nd, ns = w.shape
        b = attention_mask(nd, ns)
        b = b.view(1, 1, *b.shape) # ensure mask will broadcast to batch, heads dimensions.
        m = w*b - 1e10 * (1-b)
        return m

    def forward(self, x, past=None):
        """
        :param x: transformer layer input tensor of shape [B, T2, d_model]
        :param past: optional past key value tensor of shape [B, 2, H, T1, d_k]
        :return: attention output tensor of shape [B, T2, d_model]
                 and present key value tensor with shape [B, 2, H, T1+T2, d_k]
        """
        qkv = self.c_attn(x)
        qkv = tc.chunk(qkv, 3, dim=-1)
        qs, ks, vs = map(self.split_heads, qkv)  # each with shape [B, H, T, d_k]
        if past is not None:
            past_ks, past_vs = tc.unbind(past, dim=1)
            K = tc.cat((past_ks, ks), dim=-2)  # concatenate along source time axis
            V = tc.cat((past_vs, vs), dim=-2)
        present = tc.stack((ks, vs), dim=1)
        ws = tc.einsum('bhid,bhjd->bhij', qs, ks) / (self.d_k ** 0.5)
        ws = self.mask_attn_weights(ws)
        ws = tc.nn.Softmax(dim=-1)(ws)
        attn = tc.einsum('bhij,bhjd->bhid', ws, vs)
        attn = self.merge_heads(attn)  # shape is now [B, T, D].
        attn = self.c_proj(attn)
        return attn, present


class FeedForward(tc.nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.conv_stack = tc.nn.Sequential(
            tc.nn.Conv1d(d_model, d_hidden, kernel_size=(1,), stride=(1,)),
            tc.nn.GELU(),
            tc.nn.Conv1d(d_hidden, d_model, kernel_size=(1,), stride=(1,))
        )
        for m in self.conv_stack.modules():
            if isinstance(m, tc.nn.Conv1d):
                tc.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv_stack(x)


class LayerNorm(tc.nn.Module):
    def __init__(self, d_model, epsilon=1e-6):
        super().__init__()
        self.d_model = d_model
        self.gamma = tc.nn.Parameter(tc.ones(d_model))
        self.beta = tc.nn.Parameter(tc.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        mu = tc.mean(x, dim=-1, keepdim=True)
        var = tc.mean(tc.square(x-mu), dim=-1, keepdim=True)
        x = (x-mu) * tc.rsqrt(var + self.epsilon)
        view_shape = [1, 1, self.d_model]  # broadcast to BTD
        x = x * self.gamma.view(*view_shape) + self.beta.view(*view_shape)
        return x


class PreactivationTranformerLayer(tc.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_hidden=4*d_model)

    def forward(self, x, past=None):
        n1 = self.ln1(x)
        a, present = self.attn(n1, past=past)
        x = x + a

        n2 = self.ln2(x)
        f = self.ff(n2.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + f
        return x, present


class PreactivationTranformer(tc.nn.Module):
    def __init__(self, n_vocab, n_ctx, n_emb, n_heads, n_layers):
        super(PreactivationTranformer, self).__init__()
        self.n_layers = n_layers
        self.n_ctx = n_ctx

        self.token_embs = tc.nn.Embedding(n_vocab, n_emb)
        self.register_buffer('position_embs', self.position_embeddings(n_ctx+1, n_emb))
        # ^ ensures non-parameter field self.position_embs is sent to the gpu
        # when model.to('cuda') is called.
        # see https://stackoverflow.com/questions/60908827/

        self.transformer_layers = tc.nn.ModuleList([
            PreactivationTranformerLayer(
                d_model=n_emb,
                num_heads=n_heads)
            for _ in range(self.n_layers)
        ])

        self.ln_final = LayerNorm(n_emb)
        self.fc = tc.nn.Linear(n_emb, n_vocab, bias=False)

        self.init_weights()

    def init_weights(self):
        tc.nn.init.normal_(self.token_embs.weight, mean=0.0, std=0.02)
        tc.nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)

    def position_embeddings(self, n_ctx, n_emb):
        pe = tc.zeros(n_ctx, n_emb)
        position = tc.arange(0, n_ctx, dtype=tc.float).unsqueeze(1)
        div_term = tc.exp(tc.arange(0, n_emb, 2).float() * (-math.log(10000.0) / n_emb)).unsqueeze(0)
        pe[:, 0::2] = tc.sin(position * div_term)
        pe[:, 1::2] = tc.cos(position * div_term)
        return pe

    def forward(self, x, past=None):
        """
        :param x: input token longtensor of shape [B, T2]
        :param past: optional past state tensor of shape [B, L, 2, H, T1, d_k]
        :return: log probs tensor of shape [B, T2, A],
                 and new state of shape [B, L, 2, H, T1+T2, d_k].
        """
        emb_x = self.token_embs(x)
        emb_p = self.position_embs.unsqueeze(0)

        lp = 0 if past is None else past.shape[-2]
        assert lp + x.shape[1] <= self.n_ctx
        emb_p = emb_p[:, lp:lp+x.shape[1], :]

        h = emb_x + emb_p

        presents = []
        pasts = tc.unbind(past, dim=1) if past is not None else [None] * self.n_layers
        for i in range(0, self.n_layers):
            h, present = self.transformer_layers[i](h, past=pasts[i])
            presents.append(present)
        present = tc.stack(presents, dim=1)

        normed = self.ln_final(h)
        logits = self.fc(normed)

        return logits, present
