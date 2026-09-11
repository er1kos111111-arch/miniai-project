import os
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelConfig:
    vocab_size: int = 5000
    block_size: int = 128
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 2048
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    save_every: int = 500
    block_size: int = 128
    grad_clip: float = 1.0
    seed: int = 42

    data_path: str = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
    prepared_path: str = os.path.join(PROJECT_ROOT, "data", "prepared.npz")
    tokenizer_path: str = os.path.join(PROJECT_ROOT, "data", "tokenizer.json")
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")


@dataclass
class InferenceConfig:
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 150
    system_prompt: str = (
        "Ты MiniAI, маленькая дружелюбная языковая модель. "
        "Отвечай коротко, понятно и дружелюбно."
    )


@dataclass
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
