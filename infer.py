import os
import argparse
import logging
import torch
from transformers import AutoTokenizer
from src.data_loader import conll_to_segments, Token
from src.model import LawTagger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_flat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Prepare label list
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

    # 4) Prepare input segments from text or CoNLL
    if args.text:
        words = args.text.split()
        segments = [[Token(t, ["O"]) for t in words]]
    else:
        segments = conll_to_segments(args.data_file)

    # 5) Perform inference
    results = []  # list of lists of (token, label)
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

        # Map subword preds to words
        word_preds = {}
        prev_word = None
        for idx, widx in enumerate(word_ids):
            if widx is None or widx == prev_word:
                continue
            word_preds[widx] = preds[idx]
            prev_word = widx
        pred_labels = [label_list[word_preds[i]] for i in range(len(words))]

        results.append(list(zip(words, pred_labels)))

    # 6) Output: to file or stdout
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as out_f:
            for sent in results:
                for tok, lab in sent:
                    out_f.write(f"{tok} {lab}\n")
                out_f.write("\n")
        logger.info(f"Wrote predictions to {args.output_path}")
    else:
        # Print to stdout
        for sent in results:
            for tok, lab in sent:
                print(f"{tok} {lab}")
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flat NER inference (CoNLL or raw text)")
    parser.add_argument('--model_path',  required=True, help='Path to LawTagger checkpoint')
    parser.add_argument('--data_file',   required=False, help='CoNLL file to infer')
    parser.add_argument('--text',        required=False, help='Raw text to infer')
    parser.add_argument('--output_path', required=False, help='Path to write CoNLL predictions')
    parser.add_argument('--bert_model',  default='aubmindlab/bert-base-arabertv2')
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--max_length',  type=int,   default=512)
    args = parser.parse_args()

    if not args.text and not args.data_file:
        parser.error('You must specify either --text or --data_file')

    infer_flat(args)
