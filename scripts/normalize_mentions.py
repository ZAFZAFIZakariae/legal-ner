import sqlite3
from rapidfuzz import process, fuzz

def normalize_mentions(db_path="legal_ner.db", score_thresh=80):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # load all canonical codes
    c.execute("SELECT law_id, code FROM laws")
    laws = c.fetchall()  # list of (law_id, code)

    c.execute("SELECT article_id, law_id, number FROM articles")
    articles = c.fetchall()

    # get raw mentions of type LAW or ARTICLE
    c.execute("""
      SELECT mention_id, entity_text, entity_type 
      FROM mentions WHERE entity_type IN ('LAW','ARTICLE')
    """)
    for mid, text, etype in c.fetchall():
        if etype == "LAW":
            # match against law codes
            choices = {code: lid for (lid, code) in laws}
        else:
            # match against article numbers
            choices = {num: aid for (aid, _, num) in articles}

        best, score, _ = process.extractOne(
            text, 
            choices.keys(), 
            scorer=fuzz.token_sort_ratio
        )
        if score >= score_thresh:
            mapped_id = choices[best]
            col = "law_id" if etype=="LAW" else "article_id"
            c.execute(f"""
              UPDATE mentions SET {col}=? WHERE mention_id=?
            """, (mapped_id, mid))

    conn.commit()
    conn.close()
