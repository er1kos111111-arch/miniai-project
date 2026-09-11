import json
import os
import re
from collections import Counter
from typing import List, Dict


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<s>": 4,
    "</s>": 5,
    "[INST]": 6,
    "[/INST]": 7,
    "[SYS]": 8,
    "[/SYS]": 9,
}


class SimpleTokenizer:
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.vocab_size = 0

    def _split_text(self, text: str) -> List[str]:
        tokens = re.findall(r'[a-zA-Zа-яА-ЯёЁ]+|[0-9]+|[^\w\s]', text, re.UNICODE)
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
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for token in sorted(self.special_tokens.keys(), key=len, reverse=True):
                if text[i:i+len(token)] == token:
                    ids.append(self.token_to_id[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                m = re.match(r'[a-zA-Zа-яА-ЯёЁ]+|[0-9]+|[^\w\s]', text[i:], re.UNICODE)
                if m:
                    token = m.group()
                    ids.append(self.token_to_id.get(token, self.special_tokens["<unk>"]))
                    i += len(token)
                else:
                    ids.append(self.special_tokens["<unk>"])
                    i += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            if i in (self.special_tokens["<pad>"],):
                continue
            token = self.id_to_token.get(i, "<unk>")
            if token in ("<bos>", "<eos>"):
                continue
            tokens.append(token)
        text = "".join(tokens)
        text = re.sub(r'(?<=[^\s])\s+(?=[^\s])', '', text)
        return text

    def decode_no_template(self, ids: List[int]) -> str:
        skip = {
            self.special_tokens["<s>"], self.special_tokens["</s>"],
            self.special_tokens["<bos>"], self.special_tokens["<eos>"],
            self.special_tokens["<pad>"], self.special_tokens["[SYS]"],
            self.special_tokens["[/SYS]"],
        }
        tokens = []
        for i in ids:
            if i in skip:
                continue
            token = self.id_to_token.get(i, "<unk>")
            tokens.append(token)
        text = "".join(tokens)
        text = re.sub(r'\[INST\]', '', text)
        text = re.sub(r'\[/INST\]', '', text)
        return text.strip()

    def encode_chat(self, messages: List[Dict[str, str]], system_prompt: str = "") -> List[int]:
        parts = []
        parts.append("<s>[SYS]")
        parts.append(system_prompt)
        parts.append("[/SYS]")
        for msg in messages:
            if msg["role"] == "user":
                parts.append("[INST]")
                parts.append(msg["content"])
                parts.append("[/INST]")
            elif msg["role"] == "assistant":
                parts.append(msg["content"])
                parts.append("</s>")
        full_text = "".join(parts)
        return self.encode(full_text)

    def encode_chat_prompt(self, messages: List[Dict[str, str]], system_prompt: str = "") -> List[int]:
        parts = []
        parts.append("<s>[SYS]")
        parts.append(system_prompt)
        parts.append("[/SYS]")
        for msg in messages:
            if msg["role"] == "user":
                parts.append("[INST]")
                parts.append(msg["content"])
                parts.append("[/INST]")
        full_text = "".join(parts)
        return self.encode(full_text)

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
