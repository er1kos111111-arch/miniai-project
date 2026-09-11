import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from inference.chat import load_model, generate_response
from model.config import PROJECT_ROOT, InferenceConfig


device = None
model = None
tokenizer = None
mc = None
ic = InferenceConfig()


def init_model():
    global device, model, tokenizer, mc
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "model_final.pt")
    tokenizer_path = os.path.join(PROJECT_ROOT, "data", "tokenizer.json")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No checkpoint found. Run training first: python training/train.py")
    model, tokenizer, mc = load_model(checkpoint_path, tokenizer_path, device)


def chat_fn(message, history, temperature, max_tokens):
    global model, tokenizer, mc, device
    if model is None:
        init_model()

    if not message.strip():
        return history, ""

    messages = []
    for user_msg, ai_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if ai_msg:
            messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": message})

    response = generate_response(
        model, tokenizer, mc, messages, device,
        system_prompt=ic.system_prompt,
        temperature=temperature,
        max_new_tokens=int(max_tokens)
    )

    if not response:
        response = "Я пока не знаю, что ответить. Попробуй перефразировать!"

    history.append((message, response))
    return history, ""


def clear_fn():
    return [], ""


def build_ui():
    with gr.Blocks(title="MiniAI Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# MiniAI Chat\nМаленькая языковая модель для общения")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Чат", height=400)
                msg = gr.Textbox(label="Сообщение", placeholder="Напишите сообщение...")
                with gr.Row():
                    submit_btn = gr.Button("Отправить", variant="primary")
                    clear_btn = gr.Button("Очистить")
            with gr.Column(scale=1):
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                max_tokens = gr.Slider(10, 300, value=150, step=10, label="Max Tokens")
                gr.Markdown("### Команды\n- `/reset` — очистить контекст\n- `/save` — сохранить чат\n- `/quit` — выйти")

        submit_btn.click(chat_fn, [msg, chatbot, temperature, max_tokens], [chatbot, msg])
        msg.submit(chat_fn, [msg, chatbot, temperature, max_tokens], [chatbot, msg])
        clear_btn.click(clear_fn, [], [chatbot, msg])

    return demo


if __name__ == "__main__":
    print("Initializing model...")
    try:
        init_model()
        print(f"Model loaded on {device}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
