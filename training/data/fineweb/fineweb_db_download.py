import os, json, random
import pandas as pd
import duckdb
from datasets import load_dataset
from tqdm import tqdm

# ── User parameters ────────────────────────────────────────────────────────────
TARGET_BYTES = 50 * 1024**3   # 50 GB
BATCH_SIZE   = 5_000          # batch size for inserts
OUTPUT_DB    = "training/data/fineweb/fineweb.db"
# ────────────────────────────────────────────────────────────────────────────────

# 1) Ensure output dir exists
os.makedirs(os.path.dirname(OUTPUT_DB), exist_ok=True)

# 2) Connect & create table
conn = duckdb.connect(OUTPUT_DB)
conn.execute("PRAGMA threads=8;")
conn.execute("""
  CREATE TABLE IF NOT EXISTS fineweb (
    id             VARCHAR,
    dump           VARCHAR,
    url            VARCHAR,
    file_path      VARCHAR,
    date           VARCHAR,
    language       VARCHAR,
    language_score DOUBLE,
    token_count    BIGINT,
    text           VARCHAR
  );
""")

# 3) Stream the dataset (default config, all dumps)
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

buffer = []
bytes_on_disk = os.path.getsize(OUTPUT_DB)

for ex in tqdm(ds, desc="Downloading to DuckDB"):
    buffer.append({
        "id":             ex["id"],
        "dump":           ex["dump"],
        "url":            ex["url"],
        "file_path":      ex["file_path"],
        "date":           ex["date"],
        "language":       ex["language"],
        "language_score": ex["language_score"],
        "token_count":    ex["token_count"],
        "text":           ex["text"].replace("\n", " ").strip()
    })

    # 4) When buffer full, bulk insert via Pandas
    if len(buffer) >= BATCH_SIZE:
        df = pd.DataFrame(buffer)
        conn.register("temp_df", df)                                     # :contentReference[oaicite:0]{index=0}
        conn.execute("INSERT INTO fineweb SELECT * FROM temp_df;")
        conn.unregister("temp_df")
        buffer.clear()

        # 5) Check size & break if done
        bytes_on_disk = os.path.getsize(OUTPUT_DB)
        if bytes_on_disk >= TARGET_BYTES:
            break

# 6) Flush any remaining rows
if buffer:
    df = pd.DataFrame(buffer)
    conn.register("temp_df", df)
    conn.execute("INSERT INTO fineweb SELECT * FROM temp_df;")
    conn.unregister("temp_df")
    buffer.clear()

print(f"✅ Done: {bytes_on_disk/1024**3:.1f} GB written to {OUTPUT_DB}")
conn.close()
