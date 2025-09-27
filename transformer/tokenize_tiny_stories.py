from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast
import numpy as np
import os

# 1. Load TinyStories dataset
dataset = load_dataset("roneneldan/TinyStories", split="train[:10]")

# Make a train/val split (e.g., 90/10)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_texts = dataset["train"]["text"]
val_texts = dataset["test"]["text"]

# Train byte-level BPE with TinyStories
tokenizer = Tokenizer(models.BPE(unk_token=None))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    vocab_size=10_000,
    initial_alphabet=ByteLevel.alphabet(),
    special_tokens=["<|endoftext|>"]  # assignment's doc delimiter
)
tokenizer.train_from_iterator(train_texts, trainer)

tok = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token=None,
    bos_token=None,
    eos_token=None
)

os.makedirs("tok_tinystories", exist_ok=True)
tok.save_pretrained("tok_tinystories")

def tokenize_and_save(texts, filename):
    ids = []
    for t in texts:
        # append delimiter between docs
        ids.extend(tok.encode(t) + tok.encode("<|endoftext|>"))
    arr = np.array(ids, dtype=np.uint16)  # fits 10k vocab
    np.save(filename, arr)
    return arr

# 2. Train a BPE tokenizer
# tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
# tokenizer.train_from_iterator(train_texts, trainer)
#
# # Wrap as HF fast tokenizer for convenience
# tok = PreTrainedTokenizerFast(
#     tokenizer_object=tokenizer,
#     unk_token="<unk>",
#     pad_token="<pad>",
#     bos_token="<bos>",
#     eos_token="<eos>",
# )
#
#
# os.makedirs("tok_tinystories", exist_ok=True)
# tok.save_pretrained("tok_tinystories")
#
# # 3. Tokenize datasets and save arrays
# def tokenize_and_save(texts, filename):
#     ids = []
#     for t in texts:
#         ids.extend(tok.encode(t, add_special_tokens=True))
#     arr = np.array(ids, dtype=np.int32)
#     np.save(filename, arr)
#     return arr
#

train_ids = tokenize_and_save(train_texts, "tinystories_train.npy")
val_ids   = tokenize_and_save(val_texts, "tinystories_val.npy")

print("Train tokens:", train_ids.shape)
print("Val tokens:", val_ids.shape)

