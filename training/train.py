import os
import sys
import time
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import PROJECT_ROOT, ModelConfig, TrainConfig
from model.transformer import MiniTransformer


def load_data(prepared_path, block_size, batch_size):
    data = np.load(prepared_path)["data"]
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        if len(d) < block_size + 1:
            d = np.tile(d, (block_size + 2) // len(d) + 1)
        ix = np.random.randint(0, len(d) - block_size, size=(batch_size,))
        x = torch.from_numpy(np.stack([d[i : i + block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([d[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64))
        return x, y

    return get_batch


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters=50, device="cpu"):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        total = 0.0
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            total += loss.item()
        losses[split] = total / eval_iters
    model.train()
    return losses


def train():
    tc = TrainConfig()
    mc = ModelConfig(block_size=tc.block_size)

    os.makedirs(tc.checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    print("Loading data...")
    get_batch = load_data(tc.prepared_path, tc.block_size, tc.batch_size)

    print("Creating model...")
    model = MiniTransformer(mc)
    model = model.to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"Training for max {tc.max_steps} steps...")
    t0 = time.time()
    best_val_loss = float("inf")

    for step in range(tc.max_steps):
        lr = tc.learning_rate
        if step < tc.warmup_steps:
            lr = tc.learning_rate * (step + 1) / tc.warmup_steps
        else:
            progress = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
            lr = tc.learning_rate * 0.1 + 0.5 * tc.learning_rate * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        X, Y = get_batch("train")
        X, Y = X.to(device), Y.to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            _, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0 or step == tc.max_steps - 1:
            dt = time.time() - t0
            print(f"step {step:>5d}/{tc.max_steps} | loss {loss.item():.4f} | {dt:.1f}s")

        if step % 100 == 0 or step == tc.max_steps - 1:
            dt = time.time() - t0
            losses = estimate_loss(model, get_batch, eval_iters=5, device=device)
            print(
                f"step {step:>5d}/{tc.max_steps} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"lr {lr:.2e} | "
                f"{dt:.1f}s"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        if (step + 1) % tc.save_every == 0:
            ckpt_path = os.path.join(tc.checkpoint_dir, f"model_step{step+1}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": mc.__dict__,
                    "train_config": tc.__dict__,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(tc.checkpoint_dir, "model_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": tc.max_steps,
            "config": mc.__dict__,
            "train_config": tc.__dict__,
        },
        final_path,
    )
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    train()
