import os
import sys
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import PROJECT_ROOT, ModelConfig, InferenceConfig
from model.transformer import MiniTransformer
from model.tokenizer import SimpleTokenizer


def load_model(checkpoint_path, tokenizer_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mc = ModelConfig(**checkpoint["config"])
    model = MiniTransformer(mc)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)

    return model, tokenizer, mc


def generate_response(model, tokenizer, mc, messages, device,
                      system_prompt="", temperature=0.7, top_k=50,
                      max_new_tokens=150, top_p=0.9):
    input_ids = tokenizer.encode_chat_prompt(messages, system_prompt)
    prompt_len = len(input_ids)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    stop_token = tokenizer.special_tokens.get("</s>", None)

    generated_ids = []
    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= mc.block_size else x[:, -mc.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if stop_token is not None and idx_next.item() == stop_token:
            break

        generated_ids.append(idx_next.item())
        x = torch.cat((x, idx_next), dim=1)

    response = tokenizer.decode_no_template(generated_ids)
    response = response.strip()
    if not response:
        response = "Я пока не знаю, что ответить. Попробуй перефразировать!"
    return response


def chat():
    ic = InferenceConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "model_final.pt")
    tokenizer_path = os.path.join(PROJECT_ROOT, "data", "tokenizer.json")

    if not os.path.exists(checkpoint_path):
        print("Error: No checkpoint found. Run training first:")
        print("  python training/train.py")
        return

    print("Loading model...")
    model, tokenizer, mc = load_model(checkpoint_path, tokenizer_path, device)
    print(f"Model loaded on {device}")
    print(f"Parameters: {model.count_parameters():,}")
    print()
    print("MiniAI Chat")
    print("Commands: /reset - clear context, /save - save chat, /quit - exit")
    print()

    system_prompt = ic.system_prompt
    messages = []
    temperature = ic.temperature
    max_new_tokens = ic.max_new_tokens

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            messages = []
            print("[Context cleared]")
            continue

        if user_input.lower() == "/save":
            save_path = os.path.join(PROJECT_ROOT, "chat_log.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            print(f"[Chat saved to {save_path}]")
            continue

        if user_input.lower().startswith("/temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"[Temperature set to {temperature}]")
            except ValueError:
                print("[Invalid temperature value]")
            continue

        if user_input.lower().startswith("/maxtokens "):
            try:
                max_new_tokens = int(user_input.split()[1])
                print(f"[Max tokens set to {max_new_tokens}]")
            except ValueError:
                print("[Invalid max tokens value]")
            continue

        messages.append({"role": "user", "content": user_input})

        response = generate_response(
            model, tokenizer, mc, messages, device,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        messages.append({"role": "assistant", "content": response})

        print(f"AI: {response}")
        print()


def generate_once(user_message, chat_history=None, temperature=0.7, max_new_tokens=150):
    ic = InferenceConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "model_final.pt")
    tokenizer_path = os.path.join(PROJECT_ROOT, "data", "tokenizer.json")

    if not os.path.exists(checkpoint_path):
        return "No checkpoint found. Run training first."

    model, tokenizer, mc = load_model(checkpoint_path, tokenizer_path, device)

    messages = []
    if chat_history:
        for user_msg, ai_msg in chat_history:
            messages.append({"role": "user", "content": user_msg})
            if ai_msg:
                messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": user_message})

    response = generate_response(
        model, tokenizer, mc, messages, device,
        system_prompt=ic.system_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    return response


if __name__ == "__main__":
    chat()
