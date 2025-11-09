# train_response_generator.py
import argparse
from datasets import load_dataset
from transformers import (T5TokenizerFast, T5ForConditionalGeneration, Trainer, TrainingArguments)
import numpy as np
import evaluate
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, default="./emo-response-generator")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    return p.parse_args()

def main():
    args = parse_args()
    dataset = load_dataset("empathetic_dialogues")

    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def preprocess(examples):
        prompts = []
        targets = []
        for ctx, resp, emo in zip(examples["context"], examples["response"], examples["emotion"]):
            prefix = f"emotion: {emo} context: "
            ctx_text = " ".join(ctx) if isinstance(ctx, list) else ctx
            prompts.append(prefix + ctx_text)
            targets.append(resp)
        model_inputs = tokenizer(prompts, max_length=256, truncation=True, padding="max_length")
        labels = tokenizer(text_target=targets, max_length=64, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_steps=200,
        predict_with_generate=True
    )

    rouge = evaluate.load("rouge")
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: float(v.mid.fmeasure * 100) for k, v in result.items()}

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
    print(f"Saved response generator to {args.save_dir}")

if __name__ == "__main__":
    main()
