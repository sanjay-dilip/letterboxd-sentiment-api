# src/transformer_model.py

from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

from .config import CLEANED_DATA_PATH, MODELS_DIR
from .sentiment_model import load_training_data

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mappings
LABEL2ID: Dict[str, int] = {"negative": 0, "positive": 1}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


class LetterboxdDataset(Dataset):
    """
    Simple torch Dataset wrapper for DistilBERT fine tuning.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: DistilBertTokenizerFast,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor] = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def prepare_transformer_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Use the same balanced data as the baseline model (via load_training_data),
    then split into train/val for the transformer.
    """
    X, y_str = load_training_data()  # already balanced across classes

    y_ids = np.array([LABEL2ID[label] for label in y_str])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y_ids,
    )
    return X_train, X_val, y_train, y_val


def train_transformer_model(
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
) -> Tuple[DistilBertTokenizerFast, DistilBertForSequenceClassification]:
    """
    Fine tune DistilBERT on the balanced Letterboxd sentiment data.
    Saves model + tokenizer to models/distilbert.
    """

    print("Preparing data for DistilBERT...")
    X_train, X_val, y_train, y_val = prepare_transformer_data()

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    train_dataset = LetterboxdDataset(
        texts=list(X_train),
        labels=list(y_train),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = LetterboxdDataset(
        texts=list(X_val),
        labels=list(y_val),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print(f"Using device: {DEVICE}")
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f"Average training loss: {avg_train_loss:.4f}")

        # quick validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(DEVICE)
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        if total > 0:
            val_acc = correct / total
            print(f"Validation accuracy: {val_acc:.4f}")
        else:
            print("No validation samples available.")

        model.train()

    # Save model + tokenizer
    save_dir: Path = MODELS_DIR / "distilbert"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved DistilBERT model to {save_dir}")

    model.eval()
    return tokenizer, model


def load_transformer_model() -> Tuple[DistilBertTokenizerFast, DistilBertForSequenceClassification]:
    """
    Load a fine tuned DistilBERT model from models/distilbert.
    """
    model_dir: Path = MODELS_DIR / "distilbert"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_transformer_sentiment(
    text: str,
    tokenizer: DistilBertTokenizerFast,
    model: DistilBertForSequenceClassification,
    max_length: int = 128,
) -> dict:
    """
    Run sentiment prediction with DistilBERT for a single text.
    Returns the same shape of dict as the baseline model:
    {
      "label": ...,
      "probability": ...,
      "all_probs": {...}
    }
    """
    if not isinstance(text, str):
        text = ""

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    best_idx = int(np.argmax(probs))
    label = ID2LABEL[best_idx]

    return {
        "label": label,
        "probability": float(probs[best_idx]),
        "all_probs": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    # Example CLI entry: python -m src.transformer_model
    train_transformer_model()