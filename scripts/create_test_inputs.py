import pandas as pd
from pathlib import Path

laion_dir = "/home/almalinux/beegfs🐝/laion"
test_dir = "/home/almalinux/beegfs🐝/test_inputs"
# number of test files to create
n_files = 5   
# rows per test file                  
samples_per_file = 10          

Path(test_dir).mkdir(parents=True, exist_ok=True)

# get parquet files
files = sorted(Path(laion_dir).glob("*.parquet"))
if not files:
    print("No parquet files found in laion directory")
    exit(1)

# generate test files
for i in range(min(n_files, len(files))):
    df = pd.read_parquet(files[i])
    df_sample = df.head(samples_per_file)
    out_path = test_dir + f"/test_sample_{i}.parquet"
    df_sample.to_parquet(out_path, index=False)
    print(f"Wrote test file: {out_path}")
