import os
import clip
import faiss
import torch
import argparse
import pandas as pd
from PIL import Image
from torchvision import transforms

# detect and set device (gpu if available, else cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"

# default file and directory paths
default_index = "/home/almalinux/nfs/faiss_index/merged.index"
default_metadata = "/home/almalinux/nfs/faiss_index/merged_metadata.parquet"
default_query_dir = "/home/almalinux/nfs/test_inputs"
default_output_dir = "/home/almalinux/nfs/search_outputs"

# clip preprocessing pipeline for input images
preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])

# load and preprocess a single image
def load_image(path):
    try:
        return preprocess(Image.open(path).convert("RGB"))
    
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        
        return None

# embed image using clip model
def embed_image(path, model):
    img = load_image(path)
    
    if img is None:
        return None
    
    with torch.no_grad():
        z = model.encode_image(img.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()
        
    return z.astype("float32")

# search the faiss index for similar vectors
def search(index_path, metadata_path, query_vec, top_k):
    index = faiss.read_index(index_path)
    
    # normalize query vector
    faiss.normalize_L2(query_vec.reshape(1, -1))  
    
    # search top_k results
    scores, ids = index.search(query_vec.reshape(1, -1), top_k) 
     
    metadata = pd.read_parquet(metadata_path)

    results = []
    for i in ids[0]:
        
        if i < len(metadata):
            row = metadata.iloc[i]
            results.append((row["url"], row["text"]))
            
    return results

# display results in readable format
def print_results(image_path, results):
    print(f"\n🔍 Query image: {image_path}")
    for i, (url, text) in enumerate(results):
        print(f"  {i+1:02d}. {url} | {text[:100]}...")

# determine a unique output filename to avoid overwriting
def resolve_output_filename(out_dir, image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"{base}.csv"
    full_path = os.path.join(out_dir, filename)
    i = 1
    
    while os.path.exists(full_path):
        full_path = os.path.join(out_dir, f"{base}_{i}.csv")
        i += 1
        
    return full_path

# main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_name", type=str, help="name of image in test_inputs (e.g. cat.jpg)")
    parser.add_argument("--top_k", type=int, default=5, help="number of results to return")
    parser.add_argument("--index_path", type=str, default=default_index)
    parser.add_argument("--metadata_path", type=str, default=default_metadata)
    parser.add_argument("--query_dir", type=str, default=default_query_dir)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    args = parser.parse_args()

    # construct full path to query image and prepare output directory
    image_path = os.path.join(args.query_dir, args.image_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # load clip model
    model, _ = clip.load("ViT-B/32", device=device)

    # embed query image
    query_vec = embed_image(image_path, model)
    if query_vec is None:
        exit(1)

    # perform faiss search and print results
    results = search(args.index_path, args.metadata_path, query_vec, args.top_k)
    print_results(image_path, results)

    # resolve unique output file path and save results
    out_path = resolve_output_filename(args.output_dir, image_path)
    df = pd.DataFrame(results, columns=["url", "text"])
    df.to_csv(out_path, index=False)
    
    print(f"\nResults saved to: {out_path}")
