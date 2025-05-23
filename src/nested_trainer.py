import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from src.utils import load_entity_config
from src.data_loader import parse_conll_files, get_dataloaders
from src.nested_model import NestedLawTagger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_nested(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load entity config
    entities, flat_label_list, outside = load_entity_config()

    # Load datasets
    datasets, vocabs = parse_conll_files([args.train_file, args.dev_file])
    data_config = {
        "fn": "src.datasets.NestedDataset",
        "kwargs": {"type_list": entities, "tokenizer_name": args.bert_model}
    }
    train_loader, dev_loader = get_dataloaders(
        datasets, vocabs, data_config,
        batch_size=args.batch_size
    )

    # Initialize nested model
    model = NestedLawTagger(
        bert_model_name=args.bert_model,
        label_list=flat_label_list,
        type_list=entities,
        dropout=args.dropout
    ).to(device)

    # Warm-start from flat checkpoint if provided
    if args.warmup_checkpoint:
        ckpt = torch.load(args.warmup_checkpoint, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        own_state = model.state_dict()
        loaded, ignored = model.load_state_dict(
            {k: v for k, v in state.items() if k in own_state and v.shape == own_state[k].shape},
            strict=False
        )
        logger.info(f"Warm-start: loaded {len(loaded)} tensors, ignored {len(ignored)}")

    # Optimizer, scheduler, loss
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
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Nested Epoch {epoch}/{args.epochs}", leave=False)
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            span_labels = batch['span_labels'].to(device)  # [B,L,L]

            optimizer.zero_grad()
            span_scores = model(input_ids, attention_mask)  # [B,L,L,T]
            # reshape for CrossEntropyLoss: input [B,T,L,L], target [B,L,L]
            loss = criterion(span_scores.permute(0,3,1,2), span_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / (train_bar.n + 1))

        avg_train = running_loss / len(train_loader)
        logger.info(f"[Nested] Epoch {epoch} Train Loss: {avg_train:.4f}")

        # Dev evaluation
        model.eval()
        dev_loss = 0.0
        dev_bar = tqdm(dev_loader, desc="Dev Eval", leave=False)
        with torch.no_grad():
            for batch in dev_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                span_labels = batch['span_labels'].to(device)
                span_scores = model(input_ids, attention_mask)
                loss = criterion(span_scores.permute(0,3,1,2), span_labels)
                dev_loss += loss.item()
        avg_dev = dev_loss / len(dev_loader)
        logger.info(f"[Nested] Epoch {epoch} Dev Loss: {avg_dev:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"nested_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved nested checkpoint: {ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Nested NER with warm start")
    parser.add_argument('--train_file',       required=True, help='Nested train .conll')
    parser.add_argument('--dev_file',         required=True, help='Dev .conll for nested')
    parser.add_argument('--output_dir',       required=True, help='Where to save nested checkpoints')
    parser.add_argument('--epochs',    '-e',  type=int, default=3)
    parser.add_argument('--batch_size','-b',  type=int, default=4)
    parser.add_argument('--lr',        '-l',  type=float, default=5e-5)
    parser.add_argument('--dropout',   '-d',  type=float, default=0.1)
    parser.add_argument('--bert_model',       default='aubmindlab/bert-base-arabertv2')
    parser.add_argument('--warmup_checkpoint', help='Path to flat model checkpoint for warm start')
    args = parser.parse_args()
    train_nested(args)
