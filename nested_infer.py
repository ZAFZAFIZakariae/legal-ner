import os
import argparse
import logging
import torch
from transformers import AutoTokenizer
from src.data_loader import conll_to_segments, Token
from src.nested_model import NestedLawTagger
from src.utils import load_entity_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_nested(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load entity configuration
    entities, flat_label_list, outside_tag = load_entity_config()
    num_types = len(entities)

    # Initialize model
    model = NestedLawTagger(
        bert_model_name=args.bert_model,
        label_list=flat_label_list,
        type_list=entities,
        dropout=args.dropout
    ).to(device)
    model.load(args.model_path, device)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Prepare input segments
    if args.text:
        words = args.text.split()
        # dummy gold_tag list length = num_types, not used in inference
        segments = [[Token(w, ["O"] * num_types) for w in words]]
    else:
        segments = conll_to_segments(args.data_file)

    # Inference and collect outputs
    outputs = []
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

        with torch.no_grad():
            span_scores = model(input_ids, attention_mask)[0].cpu()  # [L,L,T]

        L, _, T = span_scores.shape
        # initialize nested tags: list of T-tag lists for each token
        nested_tags = [["O"] * T for _ in range(L)]

        # assign spans above threshold
        thresh = args.threshold
        for t_idx in range(T):
            for i in range(L):
                for j in range(i, L):
                    if span_scores[i, j, t_idx] > thresh:
                        nested_tags[i][t_idx] = f"B-{entities[t_idx]}"
                        for k in range(i+1, j+1):
                            nested_tags[k][t_idx] = f"I-{entities[t_idx]}"

        outputs.append((words, nested_tags))

    # Output to file or stdout
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as out_f:
            for words, tags in outputs:
                for token, tag_list in zip(words, tags):
                    out_f.write(token + " " + " ".join(tag_list) + "\n")
                out_f.write("\n")
        logger.info(f"Wrote nested predictions to {args.output_path}")
    else:
        for words, tags in outputs:
            for token, tag_list in zip(words, tags):
                print(token, *tag_list)
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nested NER inference (CoNLL or raw text)")
    parser.add_argument("--model_path",  required=True, help="Path to nested-model checkpoint")
    parser.add_argument("--data_file",   help="CoNLL file to infer")
    parser.add_argument("--text",        help="Raw text to infer")
    parser.add_argument("--output_path", help="Path to write nested CoNLL output; if omitted prints to stdout")
    parser.add_argument("--bert_model",  default="aubmindlab/bert-base-arabertv2")
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--threshold",   type=float, default=0.0, help="Span score threshold")
    parser.add_argument("--max_length",  type=int,   default=512)
    args = parser.parse_args()

    if not (args.text or args.data_file):
        parser.error("You must specify either --text or --data_file")

    infer_nested(args)
