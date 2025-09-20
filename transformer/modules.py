import torch.nn as nn
import torch
from einops import repeat, reduce, rearrange, einsum
from einx import get_at


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
        self.register_buffer("rotation_matrix", rotation_matrix)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor):
        # required_rotation_matrix = get_at("[t] d_k_by_2 2 in, ... indices -> ... indices d_k_by_2 2 in", self.rotation_matrix, token_ids)
        # required_rotation_matrix = get_at("[t] d_k_by_2 2 in, ... -> ... d_k_by_2 2 in", self.rotation_matrix, token_ids)
        # NOTE: Lesson for why the above didn't work is in einsum don't use numbers unless in rearrange as below
        print("rotation_matrix_shape: ", self.rotation_matrix.shape)
        required_rotation_matrix = get_at("[p] n o i, ... t -> ... t n o i", self.rotation_matrix, token_ids)
        x = rearrange(x, "... t (s d2) -> ... t s d2", s=self.rotation_matrix.shape[-3], d2=2)
        x = einsum(required_rotation_matrix, x, "... t n o in, ... t n in -> ... t n o")
        x = rearrange(x, "... t n o -> ... t (n o)")

        return x
