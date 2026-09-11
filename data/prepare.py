import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import PROJECT_ROOT, ModelConfig
from model.tokenizer import SimpleTokenizer


def load_jsonl(path: str):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                continue
            messages = obj["messages"]
            valid_roles = {"system", "user", "assistant"}
            if not all(m.get("role") in valid_roles for m in messages):
                continue
            if not any(m["role"] == "user" for m in messages):
                continue
            if not any(m["role"] == "assistant" for m in messages):
                continue
            has_consecutive_assistant = False
            for i in range(1, len(messages)):
                if messages[i]["role"] == "assistant" and messages[i-1]["role"] == "assistant":
                    has_consecutive_assistant = True
                    break
            if has_consecutive_assistant:
                continue
            if any(not m.get("content", "").strip() for m in messages):
                continue
            examples.append(obj)
    return examples


def clean_dataset(examples):
    seen = set()
    cleaned = []
    for ex in examples:
        key = json.dumps([(m["role"], m["content"]) for m in ex["messages"]], ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(ex)
    return cleaned


def prepare():
    data_path = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
    prepared_path = os.path.join(PROJECT_ROOT, "data", "prepared.npz")
    tokenizer_path = os.path.join(PROJECT_ROOT, "data", "tokenizer.json")

    print("Loading dataset...")
    examples = load_jsonl(data_path)
    print(f"Loaded {len(examples)} valid examples")

    examples = clean_dataset(examples)
    print(f"After cleaning: {len(examples)} examples")

    print("Building tokenizer...")
    all_texts = []
    for ex in examples:
        for m in ex["messages"]:
            all_texts.append(m["content"])

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_texts, max_vocab=ModelConfig().vocab_size)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved: vocab_size={tokenizer.vocab_size}")

    print("Encoding training data...")
    all_ids = []
    for ex in examples:
        system_prompt = ""
        messages = []
        for m in ex["messages"]:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                messages.append(m)
        ids = tokenizer.encode_chat(messages, system_prompt)
        all_ids.extend(ids)

    print(f"Total tokens: {len(all_ids)}")

    data = np.array(all_ids, dtype=np.int32)
    np.savez(prepared_path, data=data)
    print(f"Prepared data saved to {prepared_path}")
    print("Done!")


if __name__ == "__main__":
    prepare()
