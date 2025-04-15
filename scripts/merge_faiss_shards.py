import os
import faiss
import logging
import argparse
import pandas as pd
from datetime import datetime

# setup logging to both file and stdout with timestamped filename
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"merge_faiss_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    
    # input directory with index and metadata shards
    parser.add_argument("--index_dir", type=str, required=True)  
       
    # path to save final merged faiss index         
    parser.add_argument("--output_index", type=str, required=True)    
    
    # path to save merged metadata       
    parser.add_argument("--output_metadata", type=str, required=True)
       
    # flag to normalize vectors before merging     
    parser.add_argument("--normalize", action="store_true")    
     
    # directory for log files             
    parser.add_argument("--log_dir", type=str, default="/home/almalinux/nfs/logs/merge")  
    
    return parser.parse_args()

# merge all .index shards into one faiss index
def merge_indexes(index_dir, normalize):
    index_files = sorted([f for f in os.listdir(index_dir) if f.endswith(".index")])
    if not index_files:
        logging.error("No .index files found to merge")
        raise RuntimeError("No .index files found")

    logging.info(f"Found {len(index_files)} index files")

    # read the first index as base and get dimensionality
    base_index = faiss.read_index(os.path.join(index_dir, index_files[0]))
    dim = base_index.d
    all_indexes = [base_index]

    # load remaining indexes and check dimension consistency
    for fname in index_files[1:]:
        path = os.path.join(index_dir, fname)
        idx = faiss.read_index(path)
        
        if idx.d != dim:
            logging.error(f"Dimension mismatch in {fname}")
            raise ValueError(f"Index dimension mismatch in {fname}")
        
        all_indexes.append(idx)

    # initialize a new flat inner product index for merging
    merged = faiss.IndexFlatIP(dim)

    # add all vectors from shards to merged index
    for i, idx in enumerate(all_indexes):
        # get back raw vectors
        xb = idx.reconstruct_n(0, idx.ntotal)    
            
        if normalize:
            faiss.normalize_L2(xb)
            
        merged.add(xb)
        logging.info(f"Merged shard {i+1}/{len(all_indexes)}")

    return merged

# merge all metadata parquet files into a single dataframe
def merge_metadata(index_dir):
    meta_files = sorted([f for f in os.listdir(index_dir) if f.endswith(".meta.parquet")])
    if not meta_files:
        logging.error("No .meta.parquet files found")
        raise RuntimeError("No metadata files found")

    logging.info(f"found {len(meta_files)} metadata files")
    dfs = []

    # read and collect all metadata files
    for f in meta_files:
        try:
            df = pd.read_parquet(os.path.join(index_dir, f))
            dfs.append(df)
            logging.info(f"Loaded metadata: {f} with {len(df)} rows")
        except Exception as e:
            logging.warning(f"Failed to read {f}: {e}")

    if not dfs:
        raise RuntimeError("No metadata loaded successfully")

    return pd.concat(dfs, ignore_index=True)

# main orchestration logic
def main():
    args = parse_args()
    setup_logging(args.log_dir)

    logging.info("Starting FAISS index merge process")
    merged_index = merge_indexes(args.index_dir, args.normalize)
    faiss.write_index(merged_index, args.output_index)
    logging.info(f"Saved merged index to {args.output_index}")

    logging.info("Starting metadata merge")
    merged_meta = merge_metadata(args.index_dir)
    merged_meta.to_parquet(args.output_metadata, index=False)
    logging.info(f"Saved merged metadata to {args.output_metadata}")

# run main 
if __name__ == "__main__":
    main()
