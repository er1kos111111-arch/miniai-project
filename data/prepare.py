import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import PROJECT_ROOT, ModelConfig
from model.tokenizer import SimpleTokenizer


def load_conversations(path: str):
    conversations = []
    current = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    conversations.append(current)
                    current = []
                continue
            current.append(line)
        if current:
            conversations.append(current)
    return conversations


def prepare():
    data_path = os.path.join(PROJECT_ROOT, "data", "train.txt")
    prepared_path = os.path.join(PROJECT_ROOT, "data", "prepared.npz")
    tokenizer_path = os.path.join(PROJECT_ROOT, "data", "tokenizer.json")

    print("Loading conversations...")
    conversations = load_conversations(data_path)
    print(f"Found {len(conversations)} conversations")

    texts = []
    for conv in conversations:
        full_text = "\n".join(conv)
        texts.append(full_text)

    print("Building tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts, max_vocab=ModelConfig().vocab_size)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved: vocab_size={tokenizer.vocab_size}")

    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)

    print(f"Total tokens: {len(all_ids)}")

    data = np.array(all_ids, dtype=np.int32)
    np.savez(prepared_path, data=data)
    print(f"Prepared data saved to {prepared_path}")
    print("Done!")


if __name__ == "__main__":
    prepare()
