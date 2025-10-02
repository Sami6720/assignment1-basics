from einx import get_at
from einops import rearrange, reduce, repeat
import torch
import numpy as np
import argparse
from transformer.modules import (Linear, Embedding, RMSNorm, Swiglu, RoPE, softmax, Attention, MultiHeadedCausalSelfAttention, TransformerBlock,
                                 TransformerLM, cross_entropy, AdamW, cosine_annealing_lr, gradient_clipping, get_batch, save_checkpoint, load_checkpoint)
from einops import rearrange, reduce, repeat
from einx import get_at
from transformers import PreTrainedTokenizerFast
import os
from time import time

def get_parameter_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)

def get_gradient_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm = p.grad.data.norm(norm_type)
            total_norm += grad_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)

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
            entity='doina-precup',
            project='cs-336-assignment-1',
            mode='online',
            save_code=True
        )

        model = TransformerLM(d_model, num_heads, d_ff,
                              context_length, rope_theta, vocab_size, num_layers)

        if args.get("tokenizer_dir"):
            tok = PreTrainedTokenizerFast.from_pretrained(args["tokenizer_dir"])
        # attach to model so model.generate can use it
        if tok is not None:
            model.tokenizer = tok
            model.eot_id = tok.convert_tokens_to_ids("<|endoftext|>")

        optim = AdamW(model.parameters(), config["lr"], weight_decay)

        # TODO: Need to figure out how to load the dataset.
        dataset = np.load("tinystories_train.npy", mmap_mode='r')
        validation_dataset = np.load("tinystories_val.npy", mmap_mode='r')

        best_val_loss = float('inf')

        if checkpoint_path:
            model = load_checkpoint(model, optim, checkpoint_path)

        model.to(device)
        config["training_steps_per_epoch"] = (total_tokens // batch_size) // context_length
        print("Config training steps per epoch", config["training_steps_per_epoch"])
        global_step = 0
        for i in range(config["num_epochs"]):
            for j in range(config["training_steps_per_epoch"]):
                global_step += 1
                optim.zero_grad()
                # t_b = time()
                X, Y = get_batch(dataset, context_length, batch_size, device)
                # print(f"Time taken for loading for step {j} is {time() - t_b}")
                X = X.long().to(device)
                Y = Y.long().to(device)
                X = model(X)  # B, T, V
                X = rearrange(X, "b t v -> t b v")
                Y = rearrange(Y, "b t -> t b")
                loss = cross_entropy(X, Y)
                loss.backward()

                # if config["use_gradient_clipping"]:
                    # gradient_clipping(model.parameters(), 
                optim.step()

                metric = {
                    "training_loss": loss.mean().item(),
                    "epoch": i,
                    "training_step_in_epoch": j
                }

                validate_interval = (j + 1) % config["validate_every_x_steps"] == 0
                if validate_interval:
                    X, Y = get_batch(validation_dataset, context_length, batch_size, device)
                    X = X.long().to(device)
                    Y = Y.long().to(device)
                    X = model(X)  # B, T, V
                    X = rearrange(X, "b t v -> t b v")
                    Y = rearrange(Y, "b t -> t b")
                    val_loss = cross_entropy(X, Y)
                    metric["validation_loss"] = loss.mean().item()
                    if val_loss < best_val_loss:
                        best_val = val_loss
                        best_path = os.path.join(config["ckpt_dir"], config['job_name'], "best.pt")
                        save_checkpoint(model, optim, global_step, best_path)
                        wandb.log({"ckpt/best_val": best_val}, step=global_step)
                    metric["model/param_norm"] = get_parameter_norm(model)
                    metric["model/grad_norm"] = get_gradient_norm(model)


                # ---- Periodic checkpoint
                if global_step % config["ckpt_every"] == 0:
                    path = os.path.join(config["ckpt_dir"], config['job_name'],f"step_{global_step}.pt")
                    save_checkpoint(model, optim, global_step, path)



                if (j + 1) % config["wandb_log_interval"] == 0:
                    wandb.log(metric, step=global_step)

                if ((j+1) % config["sample_every"]) == 0:
                    prompt = args.get("sample_prompt", "<|endoftext|>")
                    # you can switch strategy: 'temp_scaled_softmax' or 'top_p'
                    text = model.generate(
                        input_prompt=prompt,
                        strategy=args.get("gen_strategy", "temp_scaled_softmax"),
                        temp=args.get("temp", 0.8),
                        max_generation_len=args.get("max_new_tokens", 128),
                    )

                    wandb.log(
                        {"samples/text": wandb.Html(f"<pre>{text}</pre>"), "samples/prompt": prompt},
                        step=global_step
                    )
                    # print(f"Training step {j + 1}, sample generation: {text}")

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

    parser.add_argument("--batch_size", type=int, default=4,
                        help="Minibatch size")

    parser.add_argument("--total_tokens", type=int, default=327_680_000,
                        help="Total tokens processed (batch_size × steps × context_length).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--wandb_log_interval", type=int, default=100,
                        help="Wandb log interval")
    parser.add_argument("--validate_every_x_steps", type=int, default=1000,
                        help="How often to validate.")

    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--job_name", type=str, default="debug")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--tokenizer_dir", type=str, default="tok_tinystories")
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--sample_prompt", type=str, default="<|endoftext|>")
    parser.add_argument("--gen_strategy", type=str, default="temp_scaled_softmax", choices=["temp_scaled_softmax","top_p"])
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--ckpt_dir", type=str, default="ckpts")
    parser.add_argument("--ckpt_every", type=int, default=2000)


    args = vars(parser.parse_args())

    train(args)

