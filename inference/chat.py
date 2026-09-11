import os
import sys
import json
import torch

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


def generate_response(model, tokenizer, mc, prompt, device, temperature=0.8, top_k=40, max_new_tokens=150):
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    output = tokenizer.decode(y[0].tolist())
    generated = output[len(prompt):]
    if "\n" in generated:
        generated = generated.split("\n")[0]
    return generated.strip()


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
    chat_history = [system_prompt, ""]
    temperature = ic.temperature
    top_k = ic.top_k
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
            chat_history = [system_prompt, ""]
            print("[Context cleared]")
            continue

        if user_input.lower() == "/save":
            save_path = os.path.join(PROJECT_ROOT, "chat_log.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(chat_history))
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

        chat_history.append(f"USER: {user_input}")

        context = "\n".join(chat_history)
        response = generate_response(
            model, tokenizer, mc, context, device,
            temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens
        )

        if not response:
            response = "Я пока не знаю, что ответить. Попробуй перефразировать!"

        chat_history.append(f"ASSISTANT: {response}")
        chat_history.append("")

        print(f"AI: {response}")
        print()


def generate_once(prompt, temperature=0.8, top_k=40, max_new_tokens=150):
    ic = InferenceConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "model_final.pt")
    tokenizer_path = os.path.join(PROJECT_ROOT, "data", "tokenizer.json")

    if not os.path.exists(checkpoint_path):
        return "No checkpoint found. Run training first."

    model, tokenizer, mc = load_model(checkpoint_path, tokenizer_path, device)

    system_prompt = ic.system_prompt
    full_prompt = f"{system_prompt}\n\nUSER: {prompt}\nASSISTANT:"

    response = generate_response(
        model, tokenizer, mc, full_prompt, device,
        temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens
    )
    return response


if __name__ == "__main__":
    chat()
