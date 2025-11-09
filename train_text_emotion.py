# train_text_emotion.py
import argparse
from datasets import load_dataset, ClassLabel
from transformers import (DistilBertTokenizerFast, DistilBertForSequenceClassification,
                          Trainer, TrainingArguments)
import numpy as np
import evaluate
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, default="./emo-text-emotion")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args()
    dataset = load_dataset("go_emotions")
    # Simplify multi-label to single-label for demo: pick first label if exists
    def to_single_label(example):
        labels = example.get("labels", [])
        example["label"] = labels[0] if labels else 27  # 27 ~ neutral
        return example

    dataset = dataset.map(to_single_label)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=28)

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_steps=200
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved text emotion model to {args.save_dir}")

if __name__ == "__main__":
    main()
