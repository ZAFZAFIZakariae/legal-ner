import logging
import re
import itertools
from collections import Counter, namedtuple
from torch.utils.data import DataLoader
from src.utils import load_object

logger = logging.getLogger(__name__)
# Simple Token structure for CoNLL data
Token = namedtuple("Token", ["text", "gold_tag"])

class Vocab:
    def __init__(self, counter, specials=None):
        specials = specials or []
        self.itos = list(counter.keys()) + specials
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.word_count = counter

    def __len__(self):
        return len(self.itos)

def conll_to_segments(filename):
    """
    Convert a CoNLL-formatted file into a list of segments,
    each segment is a list of Tokens.
    """
    segments = []
    segment = []
    with open(filename, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                if segment:
                    segments.append(segment)
                    segment = []
            else:
                parts = line.split()
                token = Token(text=parts[0], gold_tag=parts[1:])
                segment.append(token)
        if segment:
            segments.append(segment)
    return segments

def tag_vocab_by_type(tags):
    """
    Build separate Vocab instances per tag type (e.g. LAW, CASE).
    """
    vocabs = []
    tag_types = sorted({tag.split("-",1)[1] for tag in tags if "-" in tag})
    for ttype in tag_types:
        pattern = re.compile(f".*-{ttype}$")
        filtered = [tag for tag in tags if pattern.match(tag)] + ["O"]
        vocabs.append(Vocab(Counter(filtered)))
    return vocabs

def parse_conll_files(data_paths):
    """
    Parse multiple CoNLL files, return datasets and vocabs.
    """
    all_datasets = []
    tokens = []
    tags = []
    for path in data_paths:
        segments = conll_to_segments(path)
        all_datasets.append(segments)
        for seg in segments:
            tokens.extend([tok.text for tok in seg])
            tags.extend(tok.gold_tag for tok in seg)
    flat_tags = list(itertools.chain(*tags))
    tag_vocabs = tag_vocab_by_type(flat_tags)
    tag_vocabs.insert(0, Vocab(Counter(flat_tags)))
    Vocabs = namedtuple("Vocabs", ["tokens","tags"])
    token_vocab = Vocab(Counter(tokens), specials=["UNK"])
    return all_datasets, Vocabs(tokens=token_vocab, tags=tag_vocabs)

def get_dataloaders(datasets, vocabs, data_config, batch_size=32, num_workers=0, shuffle=(True,False,False)):
    """
    Create DataLoader objects for each dataset split.
    """
    dataloaders = []
    for idx, examples in enumerate(datasets):
        cfg = data_config.copy()
        kwargs = cfg.get("kwargs", {})
        kwargs.update({"examples": examples, "vocab": vocabs})
        dataset = load_object(cfg["fn"], kwargs)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle[idx],
            num_workers=num_workers,
            collate_fn=dataset.collate_fn
        )
        logger.info(f"Loaded {len(loader)} batches for split {idx}")
        dataloaders.append(loader)
    return dataloaders
