import os
import argparse
import logging
import torch
from transformers import AutoTokenizer
from src.data_loader import conll_to_segments, text2segments
from src.model import LawTagger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_flat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build label list
    entities = ["LAW","CASE","COURT","JUDGE","LAWYER","COURT_CLERK","ATTORNEY_GENERAL"]
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

    # 4) Prepare segments
    if args.text:
        segments, _ = text2segments(args.text)
    else:
        segments = conll_to_segments(args.data_file)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 5) Perform inference and normalization
    with open(args.output_path, 'w', encoding='utf-8') as out_f:
        for seg in segments:
            words = [tok.text for tok in seg]
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

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()[0]

            # Map subword preds to word-level preds
            word_preds = {}
            prev_w = None
            for idx, widx in enumerate(word_ids):
                if widx is None or widx == prev_w:
                    continue
                word_preds[widx] = preds[idx]
                prev_w = widx
            pred_labels = [label_list[word_preds[i]] for i in range(len(words))]

            # Write tokens and predicted labels
            for tok, lab in zip(words, pred_labels):
                out_f.write(f"{tok} {lab}\n")
            out_f.write("\n")


    logger.info(f"Wrote predictions to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  required=True)
    parser.add_argument('--data_file',   required=False)
    parser.add_argument('--text',        type=str, help="Raw Arabic text to tag")
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--bert_model',  default='aubmindlab/bert-base-arabertv2')
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--max_length',  type=int,   default=512)
    args = parser.parse_args()
    infer_flat(args)
