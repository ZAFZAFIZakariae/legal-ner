import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Token namedtuple imported from data_loader
from src.data_loader import Token

class FlatDataset(Dataset):
    """
    Dataset for flat BIO token classification.
    Each example is a list of Token(text, gold_tag=[single_tag]).
    """
    def __init__(
        self,
        examples,
        vocab,
        tokenizer_name: str = "aubmindlab/bert-base-arabertv2",
        max_length: int = 512,
    ):
        self.examples = examples
        self.vocab = vocab            # namedtuple with tokens and tags list of Vocab per type
        # flat tag vocab is the first Vocab
        self.tag_vocab = vocab.tags[0]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        segment = self.examples[idx]
        words = [tok.text for tok in segment]
        tags = [tok.gold_tag[0] for tok in segment]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        # Align labels to wordpiece tokens
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-1)
            else:
                tag = tags[word_id]
                label_ids.append(self.tag_vocab.stoi.get(tag, self.tag_vocab.stoi.get("O")))
        labels = torch.tensor(label_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        collated = {k: torch.stack([b[k] for b in batch]) for k in keys}
        return collated




class NestedDataset(Dataset):
    """
    Dataset for nested span legal NER.
    Each example is a list of Token objects with attributes:
      - text: token string
      - gold_tag: list of BIO tags, one per entity type (ordered as in type_list)
    """
    def __init__(self, examples, vocab, type_list, tokenizer_name="aubmindlab/bert-base-arabertv2", max_length=512):
        self.examples = examples
        self.vocab = vocab            # namedtuple with .tokens and .tags (list of tag Vocab per type)
        self.type_list = type_list    # e.g. ["LAW","CASE",...]
        self.type2idx = {t: i for i, t in enumerate(self.type_list)}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        segment = self.examples[idx]
        words = [tok.text for tok in segment]
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        # Build span label matrix: shape (L, L), where L = len(words)
        L = len(words)
        span_labels = torch.full((L, L), -1, dtype=torch.long)
        # Assign diagonal spans for each entity mention
        for i, tok in enumerate(segment):
            for tag in tok.gold_tag:
                if tag.startswith("B-"):
                    ent = tag.split("-", 1)[1]
                    t_idx = self.type2idx.get(ent)
                    if t_idx is not None:
                        span_labels[i, i] = t_idx
        return {
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "span_labels": span_labels,
        }

    @staticmethod
    def collate_fn(batch):
        # 1) find max length in batch
        seq_lens = [b['input_ids'].size(0) for b in batch]
        max_len  = max(seq_lens)

        padded_input_ids, padded_masks, padded_spans = [], [], []
        for b in batch:
            ids   = b['input_ids']      # [L]
            mask  = b['attention_mask'] # [L]
            spans = b['span_labels']    # [L, L]
            L     = ids.size(0)
            pad_amt = max_len - L

            # pad input_ids & mask to [max_len]
            padded_input_ids.append(torch.cat([ids, torch.zeros(pad_amt, dtype=ids.dtype)]))
            padded_masks.append(torch.cat([mask, torch.zeros(pad_amt, dtype=mask.dtype)]))

            # create a max_len×max_len span matrix filled with -1
            pad_mat = torch.full((max_len, max_len), -1, dtype=spans.dtype)
            # use a single 2D slice into the top-left corner
            pad_mat[:L, :L] = spans

            padded_spans.append(pad_mat)

        return {
            'input_ids':      torch.stack(padded_input_ids),   # [B, max_len]
            'attention_mask': torch.stack(padded_masks),        # [B, max_len]
            'span_labels':    torch.stack(padded_spans),        # [B, max_len, max_len]
        }
