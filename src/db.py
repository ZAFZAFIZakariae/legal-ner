# src/db.py
import os
import sqlite3

DB_PATH = os.getenv('DB_PATH', 'legal_ner.db')

def get_connection():
    """Return a sqlite3 connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
