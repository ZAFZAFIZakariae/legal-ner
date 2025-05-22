import os
import argparse
import logging
import torch
from transformers import AutoTokenizer
from src.model import LawTagger
from src.data_loader import conll_to_segments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_flat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Prepare label list
    entities   = ["LAW","CASE","COURT","JUDGE","LAWYER","COURT_CLERK","ATTORNEY_GENERAL"]
    label_list = ["O"] + [f"{p}-{e}" for e in entities for p in ("B","I")]

    # 2) Load model
    model = LawTagger(
        bert_model_name=args.bert_model,
        label_list=label_list,
        dropout=args.dropout
    )
    model.load(args.model_path, device)
    model.eval()

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # 4) Read input segments
    segments = conll_to_segments(args.data_file)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 5) Open output file
    with open(args.output_path, 'w', encoding='utf-8') as out_f:
        # Iterate segments
        for seg in segments:
            words = [tok.text for tok in seg]
            # Tokenize into subwords
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.max_length
            )
            input_ids = encoding.input_ids.to(device)
            attention_mask = encoding.attention_mask.to(device)
            word_ids = encoding.word_ids(batch_index=0)

            # Predict
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()[0]

            # Map subword preds to word-level
            pred_labels = []
            prev_word_idx = None
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx == prev_word_idx:
                    continue
                label_id = preds[idx]
                pred_labels.append(label_list[label_id])
                prev_word_idx = word_idx

            # Write tokens and predictions
            for tok, lab in zip(seg, pred_labels):
                out_f.write(f"{tok.text} {lab}\n")
            out_f.write("\n")

    logger.info(f"Wrote predictions to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  required=True)
    parser.add_argument('--data_file',   required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--bert_model',  default='aubmindlab/bert-base-arabertv2')
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--max_length',  type=int,   default=512)
    args = parser.parse_args()
    infer_flat(args)
