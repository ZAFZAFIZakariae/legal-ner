# scripts/extract_mentions.py
import argparse
from src.db import get_connection
from src.data_loader import conll_to_segments

parser = argparse.ArgumentParser()
parser.add_argument('--conll', required=True, help='Nested CoNLL file')
parser.add_argument('--text_id', type=int, default=0)
args = parser.parse_args()

conn = get_connection()
cur = conn.cursor()

entities = ['LAW','CASE','COURT','JUDGE','LAWYER','COURT_CLERK','ATTORNEY_GENERAL']
segments = conll_to_segments(args.conll)

for seg_idx, seg in enumerate(segments, start=1):
    tokens = [t.text for t in seg]
    tags_per_token = [t.gold_tag for t in seg]
    for t_idx, ent in enumerate(entities):
        i = 0
        while i < len(tokens):
            if tags_per_token[i][t_idx].startswith('B-'):
                start = i
                i2 = i+1
                while i2 < len(tokens) and tags_per_token[i2][t_idx].startswith('I-'):
                    i2 += 1
                mention = ' '.join(tokens[start:i2])
                is_art = mention.startswith('المادة')
                cur.execute(
                    '''INSERT INTO mentions
                       (text_id, start_idx, end_idx, mention_text, is_article, entity_type)
                       VALUES (?,?,?,?,?,?)''',
                    (args.text_id+seg_idx, start, i2-1, mention, is_art, ent)
                )
                i = i2
            else:
                i += 1

conn.commit()
cur.close()
conn.close()
