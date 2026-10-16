from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tickets.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5


def load_data():
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["category"] != "")]
    return df


def encode_labels(df):
    unique_labels = sorted(df["category"].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df["category"].map(label2id)
    return df, label2id, id2label


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )
    return dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "macro_f1": f1["f1"]
    }


def finetune_model():
    print("Loading dataset...")
    df = load_data()

    if len(df) < 10:
        raise ValueError("Dataset too small for fine-tuning. Need at least 10 examples.")

    df, label2id, id2label = encode_labels(df)
    print(f"Labels: {label2id}")
    print(f"Total examples: {len(df)}")

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    print("Tokenizing dataset...")
    tokenized_datasets = {
        "train": tokenize_dataset(dataset["train"], tokenizer),
        "test": tokenize_dataset(dataset["test"], tokenizer)
    }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=str(MODEL_DIR / "logs"),
        logging_steps=10,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating model...")
    eval_results = trainer.evaluate()

    print(f"\nFinal Results:")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Macro F1: {eval_results['eval_macro_f1']:.4f}")

    print("Saving model and tokenizer...")
    save_path = MODEL_DIR / "transformer_classifier"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    print(f"Model saved to: {save_path}")

    return {
        "accuracy": eval_results["eval_accuracy"],
        "macro_f1": eval_results["eval_macro_f1"],
        "model_path": str(save_path),
        "label2id": label2id,
        "id2label": id2label
    }


def load_finetuned_model():
    save_path = MODEL_DIR / "transformer_classifier"

    if not save_path.exists():
        raise FileNotFoundError(
            f"No fine-tuned model found at {save_path}. "
            "Run finetune_model.py first to train a model."
        )

    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)

    return model, tokenizer


def predict_category(text, model, tokenizer):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    predicted_id = predictions.item()
    predicted_label = model.config.id2label[predicted_id]
    confidence = torch.softmax(logits, dim=-1)[0][predicted_id].item()

    return predicted_label, confidence


if __name__ == "__main__":
    finetune_model()
