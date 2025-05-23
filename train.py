import os
import logging
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from seqeval.metrics import classification_report

from src.data_loader import parse_conll_files, get_dataloaders
from src.model import LawTagger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_flat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Prepare label list
    entities = ["LAW","CASE","COURT","JUDGE","LAWYER","COURT_CLERK","ATTORNEY_GENERAL"]
    label_list = ["O"] + [f"{p}-{e}" for e in entities for p in ("B","I")]

    # 2) Load datasets
    datasets, vocabs = parse_conll_files([args.train_file, args.dev_file])
    data_cfg = {
        "fn": "src.datasets.FlatDataset",
        "kwargs": {"tokenizer_name": args.bert_model}
    }
    train_loader, dev_loader = get_dataloaders(
        datasets, vocabs, data_cfg,
        batch_size=args.batch_size
    )

    # 3) Model, optimizer, scheduler, loss
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in train_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
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

        # 5) Dev evaluation with progress bar
        model.eval()
        y_true, y_pred = [], []
        dev_bar = tqdm(dev_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in dev_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1)

                for gold_seq, pred_seq in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    true_seq = [label_list[i] for i in gold_seq if i != -1]
                    pred_seq = [label_list[i] for i in pred_seq if i != -1]
                    y_true.append(true_seq)
                    y_pred.append(pred_seq)
        report = classification_report(y_true, y_pred)
        logger.info(f"[Flat] Epoch {epoch} Dev Metrics:\n{report}")

        # 6) Save checkpoint
        ckpt = os.path.join(args.output_dir, f"flat_epoch{epoch}.pt")
        model.save(ckpt)
        logger.info(f"Saved checkpoint: {ckpt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flat NER with progress bars")
    parser.add_argument("--train_file",  required=True)
    parser.add_argument("--dev_file",    required=True)
    parser.add_argument("--output_dir",  default="output/flat")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--bert_model",  default="aubmindlab/bert-base-arabertv2")
    args = parser.parse_args()
    train_flat(args)
