import os
import argparse
import logging
import torch
from transformers import AutoTokenizer
from src.data_loader import conll_to_segments, Token
from src.model import LawTagger
from src.utils import load_entity_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_flat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load entity config
    entities, label_list, outside_tag = load_entity_config()

    # Load model
    model = LawTagger(
        bert_model_name=args.bert_model,
        label_list=label_list,
        dropout=args.dropout
    )
    model.load(args.model_path, device)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Prepare segments
    if args.text:
        words = args.text.split()
        segments = [[Token(t, [outside_tag]) for t in words]]
    else:
        segments = conll_to_segments(args.data_file)

    # Inference
    results = []
    for seg in segments:
        words = [tok.text for tok in seg]
        enc = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        word_ids = enc.word_ids(batch_index=0)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1).cpu().tolist()[0]

        # Map to word-level
        word_preds = {}
        prev = None
        for i, wid in enumerate(word_ids):
            if wid is None or wid == prev: continue
            word_preds[wid] = preds[i]
            prev = wid
        pred_labels = [label_list[word_preds[i]] for i in range(len(words))]

        results.append(list(zip(words, pred_labels)))

    # Output
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for sent in results:
                for tok, lab in sent:
                    f.write(f"{tok} {lab}\n")
                f.write("\n")
        logger.info(f"Wrote predictions to {args.output_path}")
    else:
        for sent in results:
            for tok, lab in sent:
                print(f"{tok} {lab}")
            print()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Flat NER inference (CoNLL or raw text)")
    p.add_argument("--model_path",  required=True, help="Path to flat model checkpoint")
    p.add_argument("--data_file",   help="CoNLL file to infer")
    p.add_argument("--text",        help="Raw text to infer")
    p.add_argument("--output_path", help="Where to write CoNLL output; if omitted prints to stdout")
    p.add_argument("--bert_model",  default="aubmindlab/bert-base-arabertv2")
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--max_length",  type=int,   default=512)
    args = p.parse_args()

    if not (args.text or args.data_file):
        p.error("You must specify either --text or --data_file")
    infer_flat(args)
