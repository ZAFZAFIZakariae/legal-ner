# scripts/normalize_mentions.py
import argparse
from rapidfuzz import process
from src.db import get_connection

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=int, default=80)
args = parser.parse_args()

conn = get_connection()
cur = conn.cursor()

# load canon
cur.execute('SELECT law_id, raw_text FROM laws')
laws = cur.fetchall()
law_texts = [r['raw_text'] for r in laws]

cur.execute('SELECT article_id, law_id, raw_text FROM articles')
arts = cur.fetchall()
arts_by_law = {}
for r in arts:
    arts_by_law.setdefault(r['law_id'], []).append((r['article_id'], r['raw_text']))

# normalize
cur.execute('SELECT mention_id, mention_text, is_article, entity_type, resolved_law FROM mentions')
rows = cur.fetchall()
for r in rows:
    mid, text, is_art, ent, parent_law = r
    law_id = parent_law
    score = 0
    # normalize law first
    best, score, idx = process.extractOne(text, law_texts)
    law_id = laws[idx]['law_id'] if score>=args.threshold else None

    art_id = None
    if is_art and law_id:
        arts = arts_by_law.get(law_id, [])
        texts = [a[1] for a in arts]
        best2, score2, idx2 = process.extractOne(text, texts)
        if score2>=args.threshold:
            art_id = arts[idx2][0]

    cur.execute(
        'UPDATE mentions SET resolved_law=?, resolved_article=?, match_score=? WHERE mention_id=?',
        (law_id, art_id, score, mid)
    )

conn.commit()
cur.close()
conn.close()
