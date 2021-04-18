import torch as tc

# ported from gpt-2 with love
def split_states(x, n):
    *start, m = x.shape
    return tc.reshape(x, [start, n, m//n])

def merge_states(x):
    *start, a, b = x.shape
    return tc.reshape(x, [start, a*b])

def attention_mask(nd, ns):
    return tc.ones((nd, ns)).triu()


class MaskedQKVAttention(tc.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MaskedQKVAttention, self).__init__()
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

