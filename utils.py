# utils.py
from typing import Optional, List
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]

def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def predict_image_emotion(image_path: str, top_k: int = 3):
    """Zero-shot emotion detection using CLIP: compute similarity between image and emotion text prompts."""
    model, processor = load_clip()
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    inputs = processor(text=EMOTION_LABELS, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(inputs=inputs["pixel_values"])
        text_inputs = processor(text=EMOTION_LABELS, return_tensors="pt", padding=True)
        text_features = model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
    # normalize and compute cosine
    img_feat = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    txt_feat = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    sims = (img_feat @ txt_feat.T).squeeze(0).cpu().numpy()
    idx = np.argsort(-sims)[:top_k]
    return [EMOTION_LABELS[i] for i in idx]

def combine_emotions(text_emotion: Optional[str], image_emotions: Optional[List[str]]):
    parts = []
    if text_emotion:
        parts.append(text_emotion)
    if image_emotions:
        parts.extend(image_emotions[:2])
    seen = set()
    combined = []
    for p in parts:
        if p not in seen:
            combined.append(p)
            seen.add(p)
    if not combined:
        return "neutral"
    return ", ".join(combined)
