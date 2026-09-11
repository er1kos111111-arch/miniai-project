# MiniAI — Ультра-мини языковая модель для общения

Маленькая собственная conversational LLM на базе decoder-only Transformer. Обучается за 30 минут на GPU, работает локально без внешних API.

## Характеристики

| Параметр | Значение |
|----------|----------|
| Архитектура | Decoder-only Transformer |
| Параметры | ~12M |
| Context length | 256 токенов |
| Layers | 6 |
| Attention heads | 6 |
| Hidden size | 384 |
| Требуемый VRAM | ~200 MB |

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Подготовить данные и токенизатор
python data/prepare.py

# 3. Обучить модель (~30 минут на GPU)
python training/train.py

# 4. Запустить чат в терминале
python inference/chat.py

# 5. Или запустить веб-интерфейс
python app.py
```

## Структура проекта

```
project/
├── model/
│   ├── transformer.py   # Архитектура модели
│   ├── config.py        # Конфигурации
│   └── tokenizer.py     # Простой токенизатор
├── data/
│   ├── train.txt        # Датасет для обучения
│   └── prepare.py       # Подготовка данных
├── training/
│   └── train.py         # Скрипт обучения
├── inference/
│   └── chat.py          # Интерактивный чат
├── checkpoints/         # Сохранённые модели
├── app.py               # Gradio веб-интерфейс
├── requirements.txt
└── README.md
```

## Как изменить размер модели

В файле `model/config.py` измените параметры:

```python
@dataclass
class ModelConfig:
    n_layer: int = 6      # Количество слоёв (6-8)
    n_head: int = 6       # Количество голов внимания (6-8)
    n_embd: int = 384     # Размер скрытого слоя (256-512)
    block_size: int = 256 # Длина контекста (256-512)
```

Увеличение этих параметров пропорционально увеличивает модель.

## Как заменить dataset

Отредактируйте файл `data/train.txt`. Формат:

```
USER: Привет
ASSISTANT: Привет! Как дела?

USER: Как тебя зовут?
ASSISTANT: Я MiniAI.
```

Пустые строки разделяют диалоги. Затем перезапустите подготовку:

```bash
python data/prepare.py
python training/train.py
```

## Как продолжить обучение

Добавьте в `training/train.py` загрузку чекпоинта и увеличьте `max_steps`. Или просто запустите обучение заново — модель переобучится на тех же данных с нуля.

## Где находится checkpoint

Файлы моделей сохраняются в папку `checkpoints/`:
- `model_final.pt` — финальная модель
- `model_step{N}.pt` — промежуточные чекпоинты (каждые 500 шагов)

## Терминальные команды чата

- `/reset` — очистить контекст диалога
- `/save` — сохранить историю чата в файл
- `/quit` — выйти
- `/temp 0.5` — установить temperature
- `/maxtokens 200` — установить максимальное количество токенов

## Лицензия

MIT
