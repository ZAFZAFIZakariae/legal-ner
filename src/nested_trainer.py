import os
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.data_loader import parse_conll_files, get_dataloaders
from src.nested_model import NestedLawTagger

logger = logging.getLogger(__name__)


def train_nested(
    train_file: str,
    dev_file: str,
    entities: list,
    bert_model_name: str = "aubmindlab/bert-base-arabertv2",
    output_dir: str = "output/nested",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 5e-5,
    device: str = None,
):
    """
    Training loop for nested legal NER using NestedLawTagger.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build label_list and type_list
    label_list = ["O"] + [f"{p}-{e}" for e in entities for p in ("B", "I")]
    type_list = entities

    # Load datasets and vocabs
    datasets, vocabs = parse_conll_files([train_file, dev_file])
    data_config = {
    "fn": "src.datasets.NestedDataset",
    "kwargs": {"type_list": entities, "tokenizer_name": bert_model_name}
}

    train_loader, dev_loader = get_dataloaders(datasets, vocabs, data_config, batch_size=batch_size)

    # Initialize model
    model = NestedLawTagger(
        bert_model_name=bert_model_name,
        label_list=label_list,
        type_list=type_list,
        dropout=0.1,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            span_labels = batch.get("span_labels").to(device)  # shape: [B, S, S]

            optimizer.zero_grad()
            span_scores = model(input_ids, attention_mask)  # [B, S, S, T]

            B, S, _, T = span_scores.size()
            scores = span_scores.view(B * S * S, T)
            labels = span_labels.view(B * S * S)

            loss = criterion(scores, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{epochs} — Train loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"nested_epoch{epoch}.pt")
        model.save(ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, type=str)
    parser.add_argument("--dev_file", required=True, type=str)
    parser.add_argument("--output_dir", default="output/nested", type=str)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    args = parser.parse_args()

    # Entities for nested extraction
    entities = [
        "LAW",
        "CASE",
        "COURT",
        "JUDGE",
        "LAWYER",
        "COURT_CLERK",
        "ATTORNEY_GENERAL",
    ]

    train_nested(
        args.train_file,
        args.dev_file,
        entities,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
