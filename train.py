import os
import logging
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report

from src.data_loader import parse_conll_files, get_dataloaders
from src.model       import LawTagger
from src.data_loader import conll_to_segments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_flat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build your label list
    entities   = ["LAW","CASE","COURT","JUDGE","LAWYER","COURT_CLERK","ATTORNEY_GENERAL"]
    label_list = ["O"] + [f"{p}-{e}" for e in entities for p in ("B","I")]

    # 2) Load CoNLL files into segments + vocabs
    datasets, vocabs = parse_conll_files([args.train_file, args.dev_file])
    data_config = {
        "fn": "src.datasets.FlatDataset",
        "kwargs": {"tokenizer_name": args.bert_model}
    }
    train_loader, dev_loader = get_dataloaders(
        datasets, vocabs, data_config,
        batch_size=args.batch_size
    )

    # 3) Model + optimizer + scheduler + loss
    model     = LawTagger(
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

    # 4) Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)   # shape: [B, L]

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)     # [B, L, C]
            B, L, C = logits.size()
            loss = criterion(logits.view(B*L, C), labels.view(B*L))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"[Flat] Epoch {epoch}/{args.epochs} — loss: {avg_loss:.4f}")

        # 5) Dev-set evaluation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                preds  = logits.argmax(dim=-1)

                # convert to lists and filter out pad (-1)
                for gold_seq, pred_seq in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    true_labels = [label_list[id] for id in gold_seq if id != -1]
                    pred_labels = [label_list[id] for id in pred_seq if id != -1]
                    y_true.append(true_labels)
                    y_pred.append(pred_labels)

        report = classification_report(y_true, y_pred)
        logger.info(f"[Flat] Dev metrics:\n{report}")

        # 6) Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"flat_epoch{epoch}.pt")
        model.save(ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",  required=True)
    parser.add_argument("--dev_file",    required=True)
    parser.add_argument("--output_dir",  default="output/flat")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--bert_model",  default="aubmindlab/bert-base-arabertv2")
    parser.add_argument("--evaluate_dev", action="store_true", help="If set, run evaluation on dev set at the end of each epoch")
    args = parser.parse_args()
    train_flat(args)
