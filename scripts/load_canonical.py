# scripts/load_canonical.py
import json
from src.db import get_connection

conn = get_connection()
cur = conn.cursor()

# Load laws
with open('data/laws_canonical.json', encoding='utf-8') as f:
    for entry in json.load(f):
        cur.execute(
            'INSERT OR IGNORE INTO laws(raw_text, normalized) VALUES(?, ?)',
            (entry['text'], entry['normalized'])
        )

# Load articles
with open('data/articles_canonical.json', encoding='utf-8') as f:
    for entry in json.load(f):
        # find law_id by raw_text
        cur.execute('SELECT law_id FROM laws WHERE raw_text=?', (entry['law_text'],))
        row = cur.fetchone()
        if row:
            law_id = row['law_id']
            cur.execute(
                'INSERT OR IGNORE INTO articles(law_id, raw_text, normalized) VALUES(?, ?, ?)',
                (law_id, entry['text'], entry['normalized'])
            )

conn.commit()
cur.close()
conn.close()
