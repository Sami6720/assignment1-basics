from einx import get_at
from einops import rearrange, reduce, repeat
import torch
import numpy as np
import argparse
from transformer.modules import (Linear, Embedding, RMSNorm, Swiglu, RoPE, softmax, Attention, MultiHeadedCausalSelfAttention, TransformerBlock,
                                 TransformerLM, cross_entropy, AdamW, cosine_annealing_lr, gradient_clipping, get_batch, save_checkpoint, load_checkpoint)
from einops import rearrange, reduce, repeat
from einx import get_at

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import numpy as np
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Vocabulary size. Typical values are in the tens to hundreds of thousands.")

    parser.add_argument("--context_length", type=int, default=256,
                        help="Sequence length (context window). Tiny datasets may not need long contexts, "
                             "but for OpenWebText you may want longer.")

    parser.add_argument("--d_model", type=int, default=512,
                        help="Transformer hidden size. Common small configs use 768, but 512 is faster.")

    parser.add_argument("--d_ff", type=int, default=1344,
                        help="Feedforward hidden size, usually ~4–8 × d_model and multiple of 64 for GPU efficiency.")

    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE θ parameter used in rotary positional embeddings.")

    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of transformer layers.")

    parser.add_argument("--num_epochs", type=int, default=4,
                        help="Num of training epochs")

    parser.add_argument("--num_heads", type=int, default=16,
                        help="Number of attention heads.")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Minibatch size")

    parser.add_argument("--total_tokens", type=int, default=327_680_000,
                        help="Total tokens processed (batch_size × steps × context_length).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--wandb_log_interval", type=int, default=100,
                        help="Wandb log interval")
    parser.add_argument("--validate_every_x_steps", type=int, default=100,
                        help="How often to validate.")

    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--job_name", type=str, default="debug")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--tokenizer_dir", type=str, default="tok_tinystories")
    parser.add_argument("--sample_every", type=int, default=30_000)
    parser.add_argument("--sample_prompt", type=str, default="<|endoftext|>")
    parser.add_argument("--gen_strategy", type=str, default="temp_scaled_softmax", choices=["temp_scaled_softmax","top_p"])
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--ckpt_dir", type=str, default="ckpts")
    parser.add_argument("--ckpt_every", type=int, default=2000)


    config = vars(parser.parse_args())


    d_model = config["d_model"]
    vocab_size = config["vocab_size"]
    context_length = config["context_length"]
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    rope_theta = config["rope_theta"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    total_tokens = config["total_tokens"]
    checkpoint_path: str = config["checkpoint_path"]
    weight_decay = config["weight_decay"]
    batch_size = config["batch_size"]
    device = config["device"]


    model = TransformerLM(d_model, num_heads, d_ff,
                          context_length, rope_theta, vocab_size, num_layers)

    if config.get("tokenizer_dir"):
        tok = PreTrainedTokenizerFast.from_pretrained(config["tokenizer_dir"])
    # attach to model so model.generate can use it
    if tok is not None:
        model.tokenizer = tok
        model.eot_id = tok.convert_tokens_to_ids("<|endoftext|>")
    state_dict = torch.load(checkpoint_path)

    model.load_state_dict(state_dict, strict=False)


    model.eval()

    extra = 'last' if 'last' in checkpoint_path else 'best_val'

    gen_path = '/'.join(checkpoint_path.split('/')[:-1]) +"/"+ extra + '_' + 'gen.txt'

    with open(gen_path, 'w') as f:

        for i in range(5):
            f.write(f"Generation {i}")
            print(f"Generation {i}")
            gen = model.generate("Once upon a time", max_generation_len=200)
            print(gen)
            f.write(gen)
            f.write("\n")

