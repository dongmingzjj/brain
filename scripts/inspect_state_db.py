import sqlite3
import os

db_path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "hermes", "state.db")
conn = sqlite3.connect(db_path)

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"Database: {db_path}")
print(f"Tables: {len(tables)}\n")

for t in tables:
    name = t[0]
    print(f"=== {name} ===")
    cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
    for c in cols:
        print(f"  {c[1]:30s} {c[2]:15s} pk={c[5]}")
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  -> {count} rows")
    except:
        print(f"  -> count error")
    print()

# Sample a few rows from the main tables
for t in tables:
    name = t[0]
    try:
        rows = conn.execute(f"SELECT * FROM {name} LIMIT 2").fetchall()
        if rows:
            print(f"--- Sample from {name} ---")
            cols = [d[0] for d in conn.description]
            for row in rows:
                for col, val in zip(cols, row):
                    val_str = str(val)[:100] if val else "NULL"
                    print(f"  {col}: {val_str}")
                print()
    except Exception as e:
        print(f"  sample error: {e}")

conn.close()
