import torch.nn as nn
import torch
from einops import repeat, reduce, rearrange, einsum
from einx import get_at
from typing import Union
from jaxtyping import Float, Int
import math


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.embeddings = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim), a=-3, b=3)).to(device=device).to(dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape

        # return get_at("b [num_embeddings] d, b t 1 -> b d", repeat(self.embeddings, "n d -> b n d", b=B), rearrange(token_ids, ))
        # return get_at("[num_embedding] d, b t -> b t d", self.embeddings, token_ids)

        # NOTE: 
        #for b in B:
        #   for_batch = []
        #   for t in T:
        #       index = second_operat[b, t]
        #       item = first_operand[index, :]
        #       for_batch.append(item)
        # torch.stack(for_batch, axis=0)

        return get_at("[num_embedding] d, ... -> ... d", self.embeddings, token_ids)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = 2/(in_features + out_features)
        init_params = nn.init.trunc_normal_(torch.empty(
            out_features, in_features), std=std, a=-3 * std, b=3 * std)
        self.W = nn.Parameter(init_params).to(device=device).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "out_feat in_feat, ... in_feat -> ... out_feat")

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, dtype=None, device=None):

        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        old_dtype = x.dtype
        x = x.to(torch.float32)

        ms = reduce(torch.square(x), "b t d -> b t 1", "mean") + self.eps
        rms = torch.sqrt(ms)

        a = x / rms

        return einsum(a, self.g, "b t d, d -> b t d").to(old_dtype)


class Swiglu(nn.Module):

    def __init__(self, d_model: int, d_hidden: int, device=None):

        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.device = device
        self.w2 = Linear(d_hidden, d_model)
        self.w1 = Linear(d_model, d_hidden)
        self.w3 = Linear(d_model, d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        v1 = self.w3(x)
        gates = self.w1(x)
        gates = torch.sigmoid(gates) * gates

        v2 = einsum(v1, gates, "b t h, b t h -> b t h")

        return self.w2(v2)


class RoPE(nn.Module):


    def __init__(self, theta: float, max_seq_len: int, d_k: int, device=None):

        super().__init__()

        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

        rotation_matrix = [] # Should be of shape [max_seq_len, d_k // 2, 2, 2]
        for i in range(max_seq_len):
            along_T = []
            for j in range(d_k // 2):
                input_to_trig = torch.tensor(i / (theta ** ((2 * j)/d_k)))
                cos_x = torch.cos(input_to_trig)
                sin_x = torch.sin(input_to_trig)

                along_T.append(torch.tensor([[cos_x, -sin_x], [sin_x, cos_x]]))

            rotation_matrix.append(torch.stack(along_T, dim=0))

        rotation_matrix = torch.stack(rotation_matrix, dim=0)
        self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # required_rotation_matrix = get_at("[t] d_k_by_2 2 in, ... indices -> ... indices d_k_by_2 2 in", self.rotation_matrix, token_ids)
        # required_rotation_matrix = get_at("[t] d_k_by_2 2 in, ... -> ... d_k_by_2 2 in", self.rotation_matrix, token_ids)
        # NOTE: Lesson for why the above didn't work is in einsum don't use numbers unless in rearrange as below
        required_rotation_matrix = get_at("[p] n o i, ... t -> ... t n o i", self.rotation_matrix, token_positions)
        x = rearrange(x, "... t (s d2) -> ... t s d2", s=self.rotation_matrix.shape[-3], d2=2)
        x = einsum(required_rotation_matrix, x, "... t n o i, ... t n i -> ... t n o")
        x = rearrange(x, "... t n o -> ... t (n o)")

        return x



def softmax(x: torch.Tensor, dim: int = 0):

    m = torch.max(x, dim=dim, keepdim=True)[0]
    x -= m

    x = torch.exp(x)

    norm = torch.sum(x, dim=dim, keepdim=True)

    return  x / norm



class Attention(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=Union[torch.Tensor, None]):

        import math

        attn = einsum(K, Q, "b ... t1 d_k, b ... t2 d_k -> b ... t2 t1") * 1/math.sqrt(Q.shape[-1])

        if mask is not None:
            attn[mask == False] = float('-inf')

        attn = softmax(attn, dim=-1)

        # NOTE: How the output from the attention block depends on how many queries you had.

        return einsum(attn, V, "b ... t2 t1, b ... t1 d_v -> b ... t2 d_v")



class MultiHeadedCausalSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0
        self.d_k = d_model // num_heads # NOTE: Just by convention. Doesn't have to be this since we have out_proj matrix
        self.Q = Linear(d_model, self.d_k * num_heads)
        self.K = Linear(d_model, self.d_k * num_heads)
        self.V = Linear(d_model, self.d_k * num_heads)
        self.O = Linear(self.d_k * num_heads, d_model)
        self.attn = Attention()

        mask = torch.empty(max_seq_len, max_seq_len).fill_(1)
        mask = torch.tril(mask)
        mask = mask.bool()
        self.register_buffer("mask", mask, persistent=False)

        if theta:
            self.rope = RoPE(theta, max_seq_len=max_seq_len, d_k=self.d_k)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: Union[torch.Tensor, None] = None):
        """
        x.shape B, T, d_model
        """
        B, T, d = x.shape

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        Q = rearrange(Q, "b ... t (num_heads d_k) -> b ... num_heads t d_k", num_heads=self.num_heads, d_k=self.d_k)
        K = rearrange(K, "b ... t (num_heads d_k) -> b ... num_heads t d_k", num_heads=self.num_heads, d_k=self.d_k)

        if self.rope and token_positions is not None:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        V = rearrange(V, "b ... t (num_heads d_k) -> b ... num_heads t d_k", num_heads=self.num_heads, d_k=self.d_k)


        mask = self.mask[:T, :T]

        # mask shape needs to be b num_heads t t
        mask = repeat(mask, "t1 t2 -> b num_heads t1 t2", b=B, num_heads=self.num_heads)

        attn_out = self.attn.forward(Q, K, V, mask) # b ... num_head t d_k

        attn_out = rearrange(attn_out, "b ... num_heads t d_k -> b ... t (num_heads d_k)")

        return self.O.forward(attn_out)



class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dff: int, max_seq_len: int, theta: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff

        self.ff = Swiglu(d_model, d_hidden=dff)
        self.mha = MultiHeadedCausalSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.norm = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        token_positions = repeat(torch.arange(0, T), "t -> b t", b=B)
        x = x + self.mha(self.norm(x), token_positions)
        x = x + self.ff(self.norm2(x))

        return x


class TransformerLM(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dff: int, max_seq_len: int, theta: int, vocab_size: int, num_layers: int):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.num_layers = num_layers


        self.embed = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dff, max_seq_len, theta)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.out_embed = Linear(d_model, vocab_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T = x.shape
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.out_embed(x)

        # x = softmax(x, dim=-1) #NOTE: softmax is done in the loss
        return x



def cross_entropy(logits: Float[torch.Tensor, "... b v"], targets: Int[torch.Tensor, "... b"]):

    # NOTE: The following results in neumerical instablity giving rise nan values.

    # probs = softmax(logits, dim=-1)
    # probs = get_at("... b [v], ... b -> ... b", probs, targets)
    # loss = torch.mean(torch.log(probs) * -1)

    logits -= logits.max(dim=-1, keepdim=True)[0]
    e_logits = torch.exp(logits)
    sum_exp = reduce(e_logits, "... b v -> ... b", "sum")
    log_sum_exp = torch.log(sum_exp)
    targets_ei = get_at("... b [v], ... b -> ... b", logits, targets)

    return torch.mean(-1 * (targets_ei -  log_sum_exp))



class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr, weight_decay, eps=10e-8, betas=(0.9, 0.999)):

        defaults = {"lr": lr}
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay

        super().__init__(params, defaults)
        #
        # for group in self.param_groups:
        #
        #     for p in group["params"]:
        #
        #         self.state[p]["m"] = torch.zeros_like(p.data)
        #         self.state[p]["t"] = 1
                # self.state[p]["v"] = torch.zeros_like(p)


    def step(self, closure=None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group["params"]:
                p: nn.Parameter
                if not p.requires_grad:
                    continue

                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)

                g = p.grad.data

                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * torch.square(g)

                step_size = self.lr * math.sqrt(1 - self.beta2 ** t)/ (1 - self.beta1 ** t)
                update = m / (torch.sqrt(v) + self.eps)
                p.data -= step_size * update
                p.data -= self.lr * self.weight_decay * p.data

                self.state[p]["m"] = m
                self.state[p]["v"] = v
                self.state[p]["t"] = t + 1

        return loss


