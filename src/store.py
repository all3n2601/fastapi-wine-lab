"""SQLite storage for prediction history and short-lived background jobs."""
import json
import os
import sqlite3
from pathlib import Path
from src.data import ROOT

DB_PATH = Path(os.environ.get("WINE_DB_PATH", str(ROOT / "runtime/history.sqlite3")))

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection

def initialize():
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            cultivar TEXT NOT NULL, predicted_class INTEGER NOT NULL,
            input TEXT NOT NULL, output TEXT NOT NULL, note TEXT NOT NULL DEFAULT '')""")
        db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL, result TEXT, error TEXT)""")
        # Jobs interrupted by a service restart cannot finish in this process.
        db.execute("UPDATE jobs SET status='failed', error='Service restarted before completion' "
                   "WHERE status IN ('queued','running')")

def record(payload, output):
    with connect() as db:
        cursor = db.execute("""INSERT INTO predictions
            (created_at,cultivar,predicted_class,input,output)
            VALUES (strftime('%Y-%m-%dT%H:%M:%fZ','now'),?,?,?,?)""",
            (output["cultivar"], output["predicted_class"], json.dumps(payload), json.dumps(output)))
        return cursor.lastrowid

def unpack(row):
    value = dict(row)
    value["input"] = json.loads(value["input"])
    value["output"] = json.loads(value["output"])
    return value
