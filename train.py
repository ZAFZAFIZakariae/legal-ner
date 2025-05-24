import os
import logging
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from seqeval.metrics import classification_report
import warnings
from seqeval.metrics.v1 import UndefinedMetricWarning

from src.utils import load_entity_config
from src.data_loader import parse_conll_files, get_dataloaders
from src.model import LawTagger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_flat(args):
    # Silence seqeval undefined-metric warnings
    warnings.filterwarnings(
        "ignore",
        category=UndefinedMetricWarning,
        module="seqeval.metrics.v1"
    )

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load entity configuration
    entities, label_list, outside_tag = load_entity_config()

    # Parse datasets and build dataloaders
    datasets, vocabs = parse_conll_files([args.train_file, args.dev_file])
    data_config = {
        "fn": "src.datasets.FlatDataset",
        "kwargs": {"tokenizer_name": args.bert_model}
    }
    train_loader, dev_loader = get_dataloaders(
        datasets, vocabs, data_config,
        batch_size=args.batch_size
    )

    # Initialize model, optimizer, scheduler, loss
    model = LawTagger(
        bert_model_name=args.bert_model,
        label_list=label_list,
        dropout=args.dropout
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in train_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)  # [B, L, C]
            B, L, C = logits.size()
            loss = criterion(logits.view(B*L, C), labels.view(B*L))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / (train_bar.n + 1))

        avg_train_loss = running_loss / len(train_loader)
        logger.info(f"[Flat] Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # Evaluation on dev set
        model.eval()
        y_true, y_pred = [], []
        dev_bar = tqdm(dev_loader, desc="Evaluating", leave=True)
        with torch.no_grad():
            for batch in dev_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1)

                # Filter out padding (-1) positions for both gold and pred
                for gold_seq, pred_seq in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    true_seq_filtered = []
                    pred_seq_filtered = []
                    for g, p in zip(gold_seq, pred_seq):
                        if g == -1:
                            continue
                        true_seq_filtered.append(label_list[g])
                        pred_seq_filtered.append(label_list[p])
                    y_true.append(true_seq_filtered)
                    y_pred.append(pred_seq_filtered)

        report = classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
        logger.info(f"[Flat] Epoch {epoch} Dev Metrics:
{report}")
        print(f"[Flat] Epoch {epoch} Dev Metrics:
{report}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"flat_epoch{epoch}.pt")
        model.save(ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flat NER with progress bars")
    parser.add_argument("--train_file",  required=True, help="Path to train .conll file")
    parser.add_argument("--dev_file",    required=True, help="Path to dev .conll file")
    parser.add_argument("--output_dir",  default="output/flat", help="Dir to save checkpoints")
    parser.add_argument("--epochs",      type=int,   default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size",  type=int,   default=8, help="Batch size")
    parser.add_argument("--lr",          type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--dropout",     type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--bert_model",  default="aubmindlab/bert-base-arabertv2", help="Pretrained BERT model")
    args = parser.parse_args()
    train_flat(args)
