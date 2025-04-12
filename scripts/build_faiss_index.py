import os
import faiss
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# get slurm array task id for distributed indexing
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

# load embeddings and metadata from parquet file
def load_embeddings(path):
    df = pd.read_parquet(path)
    x = np.stack(df["embedding"].values).astype("float32")
    meta = df[["sample_id", "url", "text"]]
    return x, meta

# create faiss index from vector matrix
def build_index(x, normalize):
    if normalize:
        faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    return index

# save faiss index and corresponding metadata
def save_outputs(index, meta, output_dir, base_name, prefix):
    index_file = os.path.join(output_dir, f"{prefix}_{base_name}.index")
    meta_file = os.path.join(output_dir, f"{prefix}_{base_name}.meta.parquet")
    faiss.write_index(index, index_file)
    meta.to_parquet(meta_file, index=False)

# main
def main(input_dir, output_dir, prefix, normalize):
    # list all embedding parquet files
    files = sorted(Path(input_dir).glob("*.parquet"))

    # validate task id
    if task_id < 0 or task_id >= len(files):
        print(f"Invalid slurm array task id: {task_id}")
        return

    # select file based on task id
    file_path = files[task_id]
    base_name = file_path.stem

    # process file
    x, meta = load_embeddings(file_path)
    index = build_index(x, normalize)
    save_outputs(index, meta, output_dir, base_name, prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # directory with embedding parquets
    parser.add_argument("--input_dir", type=str, required=True) 
     
    # directory to save index and metadata        
    parser.add_argument("--output_dir", type=str, required=True)  
    
    # prefix for output file names       
    parser.add_argument("--prefix", type=str, default="faiss_shard")  
    
    # whether to normalize embeddings  
    parser.add_argument("--normalize", action="store_true") 
                 
    args = parser.parse_args()
    
    # call main
    main(args.input_dir, args.output_dir, args.prefix, args.normalize)