from einx import get_at
from einops import rearrange, reduce, repeat
import torch
import numpy as np
import argparse
from transformer.modules import (Linear, Embedding, RMSNorm, Swiglu, RoPE, softmax, Attention, MultiHeadedCausalSelfAttention, TransformerBlock,
                                 TransformerLM, cross_entropy, AdamW, cosine_annealing_lr, gradient_clipping, get_batch, save_checkpoint, load_checkpoint)
from einops import rearrange, reduce, repeat
from einx import get_at


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

    parser.add_argument("--num_heads", type=int, default=16,
                        help="Number of attention heads.")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Minibatch size")

    parser.add_argument("--total_tokens", type=int, default=327_680_000,
                        help="Total tokens processed (batch_size × steps × context_length).")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()


    def train(config):

        d_model = config["d_model"]
        vocab_size = config["vocab_size"]
        context_length = config["context_length"]
        d_model = config["d_model"]
        d_ff = config["d_ff"]
        rope_theta = config["rope_theta"]
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        total_tokens = config["total_tokens"]
        checkpoint_path = config["checkpoint_path"]
        weight_decay = config["weight_decay"]
        batch_size = config["batch_size"]
        device = config["device"]

        import wandb
        wandb.init(
            name=config["job_name"],
            config=config,
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            mode=config['wandb_mode'],
            save_code=True
        )

        model = TransformerLM(d_model, num_heads, d_ff,
                              context_length, rope_theta, vocab_size, num_layers)
        optim = AdamW(model.parameters(), weight_decay)

        # TODO: Need to figure out how to load the dataset.
        dataset = ...
        validation_dataset = ...




        if checkpoint_path:
            model = load_checkpoint(model, optim, checkpoint_path)

        for i in range(config["num_epochs"]):
            for j in range(config["training_steps_per_epoch"]):
                optim.zero_grad()
                X, Y = get_batch(dataset, context_length, batch_size, device)
                X = model(X)  # B, T, V
                X = rearrange(X, "b t v -> t b v")
                Y = rearrange(Y, "b t -> t b")
                loss = cross_entropy(model(X), Y)
                loss.backward()
                optim.step()

                metric = {
                    "training_loss": loss.mean().item(),
                    "epoch": i,
                    "training_step_in_epoch": j
                }

                validate_interval = j % config["validate_every_x_steps"]
                if validate_interval:
                    X, Y = get_batch(validation_dataset, context_length, batch_size, device)
                    X = model(X)  # B, T, V
                    X = rearrange(X, "b t v -> t b v")
                    Y = rearrange(Y, "b t -> t b")
                    loss = cross_entropy(model(X), Y)
                    metric["validation_loss"] = loss.mean().item()
