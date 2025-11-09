# inference.py
from transformers import pipeline, T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer, AutoModelForSequenceClassification
from utils import predict_image_emotion, combine_emotions
import torch

class EmoAssistant:
    def __init__(self, text_emotion_model: str, response_model: str, device: int = None):
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        # text emotion pipeline (single-label)
        self.text_clf = pipeline("text-classification", model=text_emotion_model, device=self.device, return_all_scores=False)
        # response generator
        self.response_tokenizer = T5TokenizerFast.from_pretrained(response_model)
        self.response_model = T5ForConditionalGeneration.from_pretrained(response_model).to("cuda" if torch.cuda.is_available() else "cpu")

    def detect_text_emotion(self, text: str):
        res = self.text_clf(text)[0]
        return res.get("label", "neutral")

    def respond(self, user_text: str, image_path: str = None, max_length: int = 64):
        text_emotion = self.detect_text_emotion(user_text)
        image_emotions = None
        if image_path:
            image_emotions = predict_image_emotion(image_path)
        combined = combine_emotions(text_emotion, image_emotions)
        prompt = f"emotion: {combined} context: {user_text}"
        inputs = self.response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.response_model.device)
        outputs = self.response_model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
        reply = self.response_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reply

if __name__ == "__main__":
    assistant = EmoAssistant(text_emotion_model="distilbert-base-uncased", response_model="t5-small")
    print(assistant.respond("I failed my exam today and feel terrible."))
