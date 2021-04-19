import torch as tc
import math

# ported from gpt-2 with love
def split_states(x, n):
    *start, m = x.shape
    return tc.reshape(x, [start, n, m//n])

def merge_states(x):
    *start, a, b = x.shape
    return tc.reshape(x, [start, a*b])

def attention_mask(nd, ns):
    # this assumes nd = ns. during generation, this won't hold, since we generate one token at a time.
    # the ns from openai includes any additional 'past' tokens and their resulting state,
    # which appears to be required to generate correctly. FIX LATER.
    return tc.ones((nd, ns)).tril()


class MultiheadAttention(tc.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
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
        b = tc.reshape(b, [1, 1] + b.shape)
        m = w*b - 1e10 * (1-b)
        return m

    def forward(self, x):
        QKV = self.c_attn(x)
        Q, K, V = map(self.split_heads, tc.split(QKV, 3, dim=-1)) # split Q, K, V and organize each as [B, H, T, d_k].
        w = tc.einsum('bhid,bhjd->bhij', Q, K) * tc.rsqrt(self.d_k)
        w = self.mask_attn_weights(w)
        w = tc.nn.Softmax(dim=-1)(w)
        a = tc.einsum('bhij,bhjd->bhid', w, V)
        a = self.merge_heads(a) # shape is now [B, T, D].
        a = self.c_proj(a)
        return a


class FeedForward(tc.nn.Module):
    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.conv_stack = tc.nn.Sequential(
            tc.nn.Conv1d(d_model, d_hidden, (1,1), stride=(1,1)),
            tc.nn.GELU(),
            tc.nn.Conv1d(d_hidden, d_model, (1,1), stride=(1,1))
        )
        for m in self.conv_stack.modules():
            if isinstance(m, tc.nn.Conv1d):
                tc.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv_stack(x)


class LayerNorm(tc.nn.Module):
    def __init__(self, d_model, epsilon=1e-6):
        self.gamma_vec = tc.nn.Parameter(tc.tensor(data=tc.ones(d_model)))
        self.beta_vec = tc.nn.Parameter(tc.tensor(data=tc.zeros(d_model)))
        self.epsilon = epsilon

    def forward(self, x):
        mu = tc.mean(x, dim=-1, keepdim=True)
        var = tc.mean(tc.square(x-mu), dim=-1, keepdim=True)
        x = (x-mu) * tc.rsqrt(var + self.epsilon)
        #broadcast_shape = [-1] + [1]*x.shape[:-1] + [self.d_model]
        #gamma = tc.reshape(self.gamma_vec, broadcast_shape)
        #beta = tc.reshape(self.beta_vec, broadcast_shape)
        #x = gamma * x + beta
        x = self.gamma_vec * x + self.beta_vec
        return x


class PreactivationTranformerLayer(tc.nn.Module):
    def __init__(self, d_model, num_heads):
        super(PreactivationTranformerLayer, self).__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_hidden=4*d_model)

    def forward(self, x):
        n1 = self.ln1(x)
        a = self.attn(n1)
        x = x + a

        n2 = self.ln2(x)
        f = self.ff(n2)
        x = x + f
        return x


class PreactivationTranformer(tc.nn.Module):
    def __init__(self, n_vocab, n_ctx, n_emb, n_heads, n_layers):
        super(PreactivationTranformer, self).__init__()
        self.n_layers = n_layers

        self.token_embedder = tc.nn.Embedding(n_vocab, n_emb)
        tc.nn.init.normal_(self.token_embedder.weight, mean=0.0, std=0.02)

        self.position_emb_mat = self.position_embeddings(n_ctx, n_emb)
        self.register_buffer('position_emb_mat', self.position_emb_mat)
        #^ afaik, if not parameters, these wont be sent to gpu unless described as buffers

        self.transformer_stack = tc.nn.ModuleList()
        for i in range(n_layers):
            self.transformer_stack.append(PreactivationTranformerLayer(d_model=n_emb, num_heads=n_heads))

        self.ln_final = LayerNorm(n_emb)
        self.fc = tc.nn.Linear(n_emb, n_vocab, bias=False)

    def position_embeddings(self, n_ctx, n_emb):
        pe = tc.zeros(n_ctx, n_emb)
        position = tc.arange(0, n_ctx, dtype=tc.float).unsqueeze(1)
        div_term = tc.exp(tc.arange(0, n_emb, 2).float() * (-math.log(10000.0) / n_emb))
        pe[:, 0::2] = tc.sin(position * div_term)
        pe[:, 1::2] = tc.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x):
        emb_x = self.token_embedder(x)
        h = emb_x + self.position_emb_mat
        for i in range(0, self.n_layers):
            h = self.transformer_stack[i](h)

        normed = self.ln_final(h)
        logits = self.fc(normed)
        return logits








