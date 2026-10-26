"""Educational Qwen2-compatible decoder; random weights only.
Architecture: Apple MLX-LM Qwen2 (MIT). Explicit attention for teaching.
"""
import mlx.core as mx
import mlx.nn as nn
CONFIG = dict(model_type='qwen2', hidden_size=256, num_hidden_layers=4, intermediate_size=768, num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-06, vocab_size=260, max_position_embeddings=512, rope_theta=1000000.0, tie_word_embeddings=True, bos_token_id=257, eos_token_id=258, pad_token_id=256)

class Attention(nn.Module):

    def __init__(self, c):
        super().__init__()
        d = c['hidden_size']
        self.h = c['num_attention_heads']
        self.k = c['num_key_value_heads']
        self.d = d // self.h
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, self.k * self.d, bias=True)
        self.v_proj = nn.Linear(d, self.k * self.d, bias=True)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.rope = nn.RoPE(self.d, traditional=False, base=c['rope_theta'])

    def __call__(self, x, cache=None):
        b, t, _ = x.shape
        offset = 0 if cache is None else cache.offset
        q = self.rope(self.q_proj(x).reshape(b, t, self.h, self.d).transpose(0, 2, 1, 3), offset=offset)
        k = self.rope(self.k_proj(x).reshape(b, t, self.k, self.d).transpose(0, 2, 1, 3), offset=offset)
        v = self.v_proj(x).reshape(b, t, self.k, self.d).transpose(0, 2, 1, 3)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        k = mx.repeat(k, self.h // self.k, axis=1)
        v = mx.repeat(v, self.h // self.k, axis=1)
        scores = q @ k.transpose(0, 1, 3, 2) * self.d ** (-0.5)
        allowed = mx.arange(k.shape[2])[None, :] <= mx.arange(t)[:, None] + offset
        attended = mx.softmax(mx.where(allowed, scores, -float('inf')), axis=-1) @ v
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(b, t, -1))

class MLP(nn.Module):

    def __init__(self, c):
        super().__init__()
        d = c['hidden_size']
        f = c['intermediate_size']
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = nn.RMSNorm(c['hidden_size'], eps=c['rms_norm_eps'])
        self.post_attention_layernorm = nn.RMSNorm(c['hidden_size'], eps=c['rms_norm_eps'])

    def __call__(self, x, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cache)
        return x + self.mlp(self.post_attention_layernorm(x))

class Body(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c['vocab_size'], c['hidden_size'])
        self.layers = [Block(c) for _ in range(c['num_hidden_layers'])]
        self.norm = nn.RMSNorm(c['hidden_size'], eps=c['rms_norm_eps'])

    def __call__(self, tokens, cache=None):
        x = self.embed_tokens(tokens)
        for i, layer in enumerate(self.layers):
            x = layer(x, None if cache is None else cache[i])
        return self.norm(x)

class Model(nn.Module):

    def __init__(self, c=CONFIG):
        super().__init__()
        self.model = Body(c)

    def __call__(self, tokens, cache=None):
        return self.model.embed_tokens.as_linear(self.model(tokens, cache))
