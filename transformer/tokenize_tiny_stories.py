from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import numpy as np
import os

# 1. Load TinyStories dataset
dataset = load_dataset("roneneldan/TinyStories", split="train[:10]")

# Make a train/val split (e.g., 90/10)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_texts = dataset["train"]["text"]
val_texts = dataset["test"]["text"]

# 2. Train a BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
tokenizer.train_from_iterator(train_texts, trainer)

# Wrap as HF fast tokenizer for convenience
tok = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
)


os.makedirs("tok_tinystories", exist_ok=True)
tok.save_pretrained("tok_tinystories")

# 3. Tokenize datasets and save arrays
def tokenize_and_save(texts, filename):
    ids = []
    for t in texts:
        ids.extend(tok.encode(t, add_special_tokens=True))
    arr = np.array(ids, dtype=np.int32)
    np.save(filename, arr)
    return arr

train_ids = tokenize_and_save(train_texts, "tinystories_train.npy")
val_ids   = tokenize_and_save(val_texts, "tinystories_val.npy")

print("Train tokens:", train_ids.shape)
print("Val tokens:", val_ids.shape)

