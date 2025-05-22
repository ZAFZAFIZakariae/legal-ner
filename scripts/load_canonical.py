import json
import psycopg2

# Adjust connection params as needed
conn = psycopg2.connect(dbname="legal", user="you", password="pw", host="localhost")
cur  = conn.cursor()

# Load and insert laws
with open("data/laws_canonical.json", encoding="utf-8") as f:
    laws = json.load(f)
for entry in laws:
    cur.execute(
        "INSERT INTO laws (raw_text, normalized) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (entry["text"], entry["normalized"])
    )

# Load and insert articles
with open("data/articles_canonical.json", encoding="utf-8") as f:
    arts = json.load(f)
for entry in arts:
    # first find law_id
    cur.execute("SELECT law_id FROM laws WHERE raw_text=%s", (entry["text"].split()[2],))
    # (assumes entry["law_text"] is exact law.raw_text)
    law_id = cur.fetchone()[0]
    cur.execute(
        "INSERT INTO articles (law_id, raw_text, normalized) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        (law_id, entry["text"], entry["normalized"])
    )

conn.commit()
cur.close()
conn.close()
