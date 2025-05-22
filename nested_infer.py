import os
import argparse
import logging
import torch
from transformers import AutoTokenizer

from src.data_loader import conll_to_segments
from src.nested_model import NestedLawTagger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_nested_conll(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define entity types and ordering (must match columns in output)
    entities = [
        "LAW", "CASE", "COURT", "JUDGE",
        "LAWYER", "COURT_CLERK", "ATTORNEY_GENERAL",
    ]

    # Instantiate model (flat labels still needed, but not used here)
    flat_labels = ["O"] + [f"{p}-{e}" for e in entities for p in ("B","I")]
    model = NestedLawTagger(
        bert_model_name=args.bert_model,
        label_list=flat_labels,
        type_list=entities,
        dropout=args.dropout
    ).to(device)
    model.load(args.model_path, device)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Prepare output
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out_f = open(args.output_path, 'w', encoding='utf-8')

    # Process each segment
    segments = conll_to_segments(args.data_file)
    for seg in segments:
        words = [tok.text for tok in seg]
        # Tokenize
        enc = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        )
        input_ids      = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        # Get span scores [1, L, L, T]
        with torch.no_grad():
            span_scores = model(input_ids, attention_mask)
        scores = span_scores[0].cpu()  # [L, L, T]

        L, _, T = scores.shape
        threshold = args.threshold

        # Initialize nested tag grid: T types x L tokens
        nested_tags = [["O"] * L for _ in range(T)]

        # Assign spans above threshold
        for t_idx in range(T):
            for i in range(L):
                for j in range(i, L):
                    if scores[i, j, t_idx] > threshold:
                        nested_tags[t_idx][i] = f"B-{entities[t_idx]}"
                        for k in range(i+1, j+1):
                            nested_tags[t_idx][k] = f"I-{entities[t_idx]}"

        # Write CoNLL: token + one column per entity type
        for idx, tok in enumerate(words):
            tags = [nested_tags[t][idx] for t in range(T)]
            out_f.write(tok + ' ' + ' '.join(tags) + '\n')
        out_f.write('\n')

    out_f.close()
    logger.info(f"Nested predictions written to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to nested-model checkpoint (e.g. output/nested/nested_epoch3.pt)")
    parser.add_argument("--data_file", required=True, help="CoNLL file with raw tokens (e.g. data/dev.conll)")
    parser.add_argument("--output_path", required=True, help="Where to write nested CoNLL predictions")
    parser.add_argument("--bert_model", default="aubmindlab/bert-base-arabertv2")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.0, help="Span logit threshold to decide entity spans")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    infer_nested_conll(args)
