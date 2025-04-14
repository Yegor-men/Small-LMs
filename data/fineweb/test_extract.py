import duckdb

DB_PATH = "training/data/fineweb/fineweb.db"

conn = duckdb.connect(DB_PATH)

query = """
SELECT text
FROM fineweb
LIMIT 3
"""

df = conn.execute(query).df()

print(df)

conn.close()
