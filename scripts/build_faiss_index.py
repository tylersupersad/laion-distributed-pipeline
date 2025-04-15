import os
import faiss
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# get slurm array task id for parallel file indexing
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

# logging helper to print and optionally save messages to a file
def log(msg, log_file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(full_msg + "\n")

# load clip embeddings and metadata from a parquet file
def load_embeddings(path):
    df = pd.read_parquet(path)
    
    # convert list of vectors to numpy array
    x = np.stack(df["embedding"].values).astype("float32")  
    # extract metadata fields
    meta = df[["sample_id", "url", "text"]]  
                   
    return x, meta

# build faiss index from numpy matrix (optionally l2-normalize)
def build_index(x, normalize):
    if normalize:
        faiss.normalize_L2(x)
    
    # use inner product for similarity
    index = faiss.IndexFlatIP(x.shape[1])  
    index.add(x)
    
    return index

# save faiss index and metadata to output directory
def save_outputs(index, meta, output_dir, base_name, prefix):
    os.makedirs(output_dir, exist_ok=True)
    index_file = os.path.join(output_dir, f"{prefix}_{base_name}.index")
    meta_file = os.path.join(output_dir, f"{prefix}_{base_name}.meta.parquet")
    faiss.write_index(index, index_file)
    meta.to_parquet(meta_file, index=False)

# main entrypoint for indexing a single parquet file
def main(input_dir, output_dir, prefix, normalize, logs_dir):
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"build_index_task{task_id}.log")

    # collect all input parquet files and validate task id
    files = sorted(Path(input_dir).glob("*.parquet"))
    if task_id < 0 or task_id >= len(files):
        log(f"[SKIP] Invalid SLURM_ARRAY_TASK_ID: {task_id}", log_path)
        exit(1)

    file_path = files[task_id]
    base_name = file_path.stem
    log(f"[TASK {task_id}] Starting FAISS index for {file_path.name}", log_path)

    try:
        # load embeddings and metadata
        x, meta = load_embeddings(file_path)
        log(f"[TASK {task_id}] Loaded embeddings: shape={x.shape}", log_path)

        # build the faiss index
        index = build_index(x, normalize)
        log(f"[TASK {task_id}] Built FAISS index (normalize={normalize})", log_path)

        # save index and metadata
        save_outputs(index, meta, output_dir, base_name, prefix)
        log(f"[TASK {task_id}] Saved index and metadata to {output_dir}", log_path)

    except Exception as e:
        log(f"[ERROR] Task {task_id} failed: {str(e)}", log_path)
        exit(1)

# parse command-line arguments and run main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # directory with parquet embedding files
    parser.add_argument("--input_dir", type=str, required=True)  
    
    # output location for faiss index files      
    parser.add_argument("--output_dir", type=str, required=True)  
    
    # prefix for saved output filenames
    parser.add_argument("--prefix", type=str, default="faiss_shard")  
    
    # enable l2 normalization before indexing
    parser.add_argument("--normalize", action="store_true")     
    
    # path for logs         
    parser.add_argument("--logs_dir", type=str, default="/home/almalinux/nfs/logs/faiss") 
     
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.prefix, args.normalize, args.logs_dir)