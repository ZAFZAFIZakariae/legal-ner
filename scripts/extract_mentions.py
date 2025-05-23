import sqlite3
import csv
from src.data_loader import conll_to_segments, Token
from src.utils import load_entity_config

def reconstruct_sentence(tokens):
    """ Join tokens with spaces and return: full_text and list of token->char_start """
    char_starts = []
    pos = 0
    parts = []
    for t in tokens:
        char_starts.append(pos)
        parts.append(t)
        pos += len(t) + 1  # +1 for the space
    text = " ".join(parts)
    return text, char_starts

def extract_mentions(nested_conll_path, db_path="legal_ner.db", text_id=1):
    # load entity types order
    entities, _, _ = load_entity_config()
    T = len(entities)

    # prepare DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    segments = conll_to_segments(nested_conll_path)  # List[List[Token]]
    for seg in segments:
        tokens = [t.text for t in seg]
        # each Token.gold_tags is a list of T BIO labels
        tags_matrix = [t.gold_tags for t in seg]

        sentence, char_starts = reconstruct_sentence(tokens)

        # 1) collect all spans per entity type
        spans = []  # (entity_type, start_tok, end_tok, entity_text)
        for t_idx, ent in enumerate(entities):
            BIO = [row[t_idx] for row in tags_matrix]
            i = 0
            while i < len(BIO):
                if BIO[i].startswith("B-"):
                    j = i
                    while j+1 < len(BIO) and BIO[j+1].startswith("I-"):
                        j += 1
                    spans.append((ent, i, j, " ".join(tokens[i:j+1])))
                    i = j+1
                else:
                    i += 1

        # 2) sort spans by length (so parents come before children if longer)
        spans.sort(key=lambda x:(x[1], - (x[2]-x[1])))  # by start, then by descending length

        # 3) insert spans, keep track of IDs for nesting
        mention_ids = []
        for ent, st, ed, etext in spans:
            start_char = char_starts[st]
            end_char   = char_starts[ed] + len(tokens[ed])
            # find parent: the last span that fully encloses this one
            parent = None
            for idx, (pent, pst, ped, _) in enumerate(spans):
                if pst <= st and ped >= ed and (pst < st or ped > ed):
                    parent = mention_ids[idx]
            cur.execute("""
              INSERT INTO mentions
                (text_id, entity_text, entity_type,
                 start_token_idx, end_token_idx, start_char, end_char, parent_id)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (text_id, etext, ent, st, ed, start_char, end_char, parent))
            mention_ids.append(cur.lastrowid)

        text_id += 1

    conn.commit()
    conn.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("nested_conll", help="Path to nested CoNLL preds")
    p.add_argument("--db", default="legal_ner.db", help="SQLite DB path")
    p.add_argument("--text_id", type=int, default=1, help="Starting text_id")
    args = p.parse_args()
    extract_mentions(args.nested_conll, args.db, args.text_id)
