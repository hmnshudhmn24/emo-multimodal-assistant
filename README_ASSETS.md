# Assets & Where to put model files

- After training `train_text_emotion.py`, save the model and tokenizer to `./emo-text-emotion/` or push to `hmnshudhmn24/emo-text-emotion`.
- After training `train_response_generator.py`, save to `./emo-response-generator/` or push to `hmnshudhmn24/emo-response-generator`.
- The inference script expects:
  - a text classifier model (HF name or local path)
  - a response generator (HF name or local path)
  - CLIP is loaded from `openai/clip-vit-base-patch32` via transformers
