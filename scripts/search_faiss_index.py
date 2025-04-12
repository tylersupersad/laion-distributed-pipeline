import os
import glob
import clip
import faiss
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# get slurm task rank and world size
rank = int(os.environ.get("SLURM_PROCID", 0))
world_size = int(os.environ.get("SLURM_NTASKS", 1))

# choose device based on availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# define the image preprocessing used by clip
preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])

# load and preprocess a local image file
def load_image(path):
    try:
        return preprocess(Image.open(path).convert("RGB"))
    except:
        return None

# embed all query images for this slurm rank
def embed_queries(paths, model):
    embeddings = []
    valid_paths = []
    
    for path in tqdm(paths, desc=f"# embedding queries on rank {rank}"):
        img = load_image(path)
        
        if img is not None:
            with torch.no_grad():
                z = model.encode_image(img.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()
            
            embeddings.append(z)
            valid_paths.append(path)
            
    return np.array(embeddings).astype("float32"), valid_paths

# search the faiss index using query vectors
def search_index(index_path, metadata_path, queries, top_k, normalize):
    # load faiss index from disk
    index = faiss.read_index(index_path)

    # optionally normalize query vectors for cosine similarity
    if normalize:
        faiss.normalize_L2(queries)

    # perform top-k search for each query
    scores, ids = index.search(queries, top_k)

    # load metadata to resolve ids back to url/text
    metadata = pd.read_parquet(metadata_path)

    # map ids to readable results
    results = []
    for id_list in ids:
        hits = []
        for i in id_list:
            if i < len(metadata):
                row = metadata.iloc[i]
                hits.append((row["url"], row["text"]))
        results.append(hits)
    return results

# print search results clearly
def display_results(query_paths, results):
    for i, hits in enumerate(results):
        print(f"\nQuery image: {query_paths[i]}")
        for j, (url, text) in enumerate(hits):
            print(f"  Rank {j+1}: {url} | {text[:100]}...")

# shard query images by rank, embed, search, display
def main(index_path, metadata_path, query_glob, top_k, normalize):
    # gather all image paths matching pattern
    all_paths = sorted(glob.glob(query_glob))

    # shard paths across workers using slurm rank
    shard_size = len(all_paths) // world_size
    start = rank * shard_size
    end = len(all_paths) if rank == world_size - 1 else (rank + 1) * shard_size
    my_paths = all_paths[start:end]

    # skip if this rank has no assigned work
    if not my_paths:
        print(f"Rank {rank} has no query images assigned")
        return

    # load clip model
    model, _ = clip.load("ViT-B/32", device=device)

    # embed queries and perform search
    embeddings, valid_paths = embed_queries(my_paths, model)
    results = search_index(index_path, metadata_path, embeddings, top_k, normalize)

    # display the results to stdout
    display_results(valid_paths, results)

# parse cli arguments and launch main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True, help="path to faiss index file")
    parser.add_argument("--metadata_path", type=str, required=True, help="path to metadata parquet file")
    parser.add_argument("--query_glob", type=str, required=True, help="glob pattern to query images")
    parser.add_argument("--top_k", type=int, default=5, help="number of results to return per query")
    parser.add_argument("--normalize", action="store_true", help="use L2 normalization on queries")
    args = parser.parse_args()
    
    # call main
    main(args.index_path, args.metadata_path, args.query_glob, args.top_k, args.normalize)