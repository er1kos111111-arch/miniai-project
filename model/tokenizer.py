import json
import os
import re
from collections import Counter
from typing import List, Dict


class SimpleTokenizer:
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.vocab_size = 0

    def _split_text(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text, re.UNICODE)
        return tokens

    def build_vocab(self, texts: List[str], max_vocab: int = 5000):
        counter = Counter()
        for text in texts:
            tokens = self._split_text(text)
            counter.update(tokens)

        self.token_to_id = dict(self.special_tokens)
        for token, _count in counter.most_common(max_vocab - len(self.special_tokens)):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        tokens = self._split_text(text)
        ids = [self.special_tokens["<bos>"]]
        for t in tokens:
            ids.append(self.token_to_id.get(t, self.special_tokens["<unk>"]))
        ids.append(self.special_tokens["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            if i in (self.special_tokens["<bos>"], self.special_tokens["<eos>"], self.special_tokens["<pad>"]):
                continue
            tokens.append(self.id_to_token.get(i, "<unk>"))
        return " ".join(tokens)

    def save(self, path: str):
        data = {
            "token_to_id": self.token_to_id,
            "vocab_size": self.vocab_size,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.vocab_size = data["vocab_size"]
