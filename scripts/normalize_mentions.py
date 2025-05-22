import psycopg2
from rapidfuzz import process

# Connect
conn = psycopg2.connect(dbname="legal", user="you", password="pw", host="localhost")
cur  = conn.cursor()

# Fetch mention records needing normalization
cur.execute("""
  SELECT mention_id, mention_text, is_article, entity_type
  FROM mentions
  WHERE resolved_law IS NULL  -- or however you mark unnormalized
""")
rows = cur.fetchall()

# Fetch canonical lists into memory
cur.execute("SELECT law_id, raw_text from laws")
laws = cur.fetchall()             # list of (law_id, raw_text)
law_texts = [r[1] for r in laws]

cur.execute("SELECT article_id, law_id, raw_text from articles")
arts = cur.fetchall()             # list of (article_id, law_id, raw_text)
arts_by_law = {}
for aid, lid, txt in arts:
    arts_by_law.setdefault(lid, []).append((aid, txt))

for mention_id, text, is_article, etype in rows:
    # 1) normalize law
    best_law, score = process.extractOne(text, law_texts)
    law_id = None
    if score > 80:
        law_id = laws[law_texts.index(best_law)][0]

    article_id = None
    if is_article and law_id:
        # only match within that law’s articles
        candidates = [a for a, txt in arts_by_law[law_id]]
        texts      = [txt for a, txt in arts_by_law[law_id]]
        best_art, art_score = process.extractOne(text, texts)
        if art_score > 80:
            article_id = candidates[texts.index(best_art)]

    # Update the mention row
    cur.execute("""
        UPDATE mentions
        SET resolved_law=%s, resolved_article=%s, match_score=%s
        WHERE mention_id=%s
    """, (law_id, article_id, score, mention_id))

conn.commit()
cur.close()
conn.close()
