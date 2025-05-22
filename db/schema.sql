-- db/schema.sql
-- Canonical laws
CREATE TABLE IF NOT EXISTS laws (
  law_id       INTEGER PRIMARY KEY,
  raw_text     TEXT    UNIQUE NOT NULL,
  normalized   TEXT    NOT NULL
);

-- Canonical articles
CREATE TABLE IF NOT EXISTS articles (
  article_id   INTEGER PRIMARY KEY,
  law_id       INTEGER NOT NULL REFERENCES laws(law_id),
  raw_text     TEXT    NOT NULL,
  normalized   TEXT    NOT NULL,
  UNIQUE (law_id, raw_text)
);

-- NER mentions
CREATE TABLE IF NOT EXISTS mentions (
  mention_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  text_id          INTEGER NOT NULL,
  start_idx        INTEGER NOT NULL,
  end_idx          INTEGER NOT NULL,
  mention_text     TEXT    NOT NULL,
  is_article       BOOLEAN NOT NULL,
  entity_type      TEXT    NOT NULL,
  resolved_law     INTEGER REFERENCES laws(law_id),
  resolved_article INTEGER REFERENCES articles(article_id),
  match_score      REAL
);
