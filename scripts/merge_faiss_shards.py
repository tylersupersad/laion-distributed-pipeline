import os
import faiss
import argparse
import pandas as pd

# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    
    # dir containing .index and .meta.parquet files
    parser.add_argument("--index_dir", type=str, required=True)  
    
    # path to save merged faiss index         
    parser.add_argument("--output_index", type=str, required=True)  
    
    # path to save merged metadata      
    parser.add_argument("--output_metadata", type=str, required=True) 
        
    # whether to l2 normalize before merging
    parser.add_argument("--normalize", action="store_true")   
                
    return parser.parse_args()

# merge all faiss index shards into a single index
def merge_indexes(index_dir, normalize):
    index_files = sorted([f for f in os.listdir(index_dir) if f.endswith(".index")])
    if not index_files:
        raise RuntimeError("No .index files found to merge")

    # load first index and capture dimension
    base_index = faiss.read_index(os.path.join(index_dir, index_files[0]))
    dim = base_index.d
    all_indexes = [base_index]

    # load remaining indexes and ensure same dimension
    for fname in index_files[1:]:
        path = os.path.join(index_dir, fname)
        idx = faiss.read_index(path)
        if idx.d != dim:
            raise ValueError(f"Index dimension mismatch in {fname}")
        all_indexes.append(idx)

    # create flat index to hold all data
    merged = faiss.IndexFlatIP(dim)
    for idx in all_indexes:
        xb = idx.reconstruct_n(0, idx.ntotal)
        merged.add(faiss.normalize_L2(xb) if normalize else xb)

    return merged

# merge all metadata files
def merge_metadata(index_dir):
    meta_files = sorted([f for f in os.listdir(index_dir) if f.endswith(".meta.parquet")])
    if not meta_files:
        raise RuntimeError("No .meta.parquet files found to merge")

    dfs = []
    for f in meta_files:
        path = os.path.join(index_dir, f)
        try:
            df = pd.read_parquet(path)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    if not dfs:
        raise RuntimeError("No metadata files could be loaded")
    
    return pd.concat(dfs, ignore_index=True)

# main
def main():
    args = parse_args()

    print("Merging faiss indexes...")
    merged_index = merge_indexes(args.index_dir, args.normalize)
    faiss.write_index(merged_index, args.output_index)
    print(f"Saved merged index: {args.output_index}")

    print("Merging metadata...")
    merged_meta = merge_metadata(args.index_dir)
    merged_meta.to_parquet(args.output_metadata, index=False)
    print(f"Saved merged metadata: {args.output_metadata}")

if __name__ == "__main__":
    main()
