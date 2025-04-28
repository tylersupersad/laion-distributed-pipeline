import os
import clip
import faiss
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

# detect and set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# default paths
default_index = "/home/almalinux/laion-distributed-pipeline/faiss_index/merged.index"
default_metadata = "/home/almalinux/laion-distributed-pipeline/faiss_index/merged_metadata.parquet"
default_query_dir = "/home/almalinux/laion-distributed-pipeline/test_inputs"
default_output_dir = "/home/almalinux/laion-distributed-pipeline/search_outputs"

# clip preprocessing
preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])

# load and preprocess image
def load_image(path):
    try:
        return preprocess(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

# embed image
def embed_image(path, model):
    img = load_image(path)
    if img is None:
        return None
    with torch.no_grad():
        z = model.encode_image(img.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()
    return z.astype("float32")

# search faiss index
def search(index_path, metadata_path, query_vec, top_k):
    index = faiss.read_index(index_path)
    faiss.normalize_L2(query_vec.reshape(1, -1))
    scores, ids = index.search(query_vec.reshape(1, -1), top_k)
    metadata = pd.read_parquet(metadata_path)
    results = []
    for i in ids[0]:
        if i < len(metadata):
            row = metadata.iloc[i]
            results.append((row["url"], row["text"]))
    return results

# print results
def print_results(image_path, results):
    print(f"\nQuery image: {os.path.basename(image_path)}")
    print("-" * 70)
    for i, (url, text) in enumerate(results):
        print(f"{i+1:02d}. URL: {url}")
        print(f"    Text: {text[:100]}...\n")
    print("-" * 70)

# resolve output filename
def resolve_output_filename(out_dir, image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"{base}.csv"
    full_path = os.path.join(out_dir, filename)
    i = 1
    while os.path.exists(full_path):
        full_path = os.path.join(out_dir, f"{base}_{i}.csv")
        i += 1
    return full_path

# prompt user for file
def select_query_image(query_dir):
    files = [f for f in os.listdir(query_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("No query images found.")
        exit(1)
    print("\nAvailable Query Images:")
    for idx, file in enumerate(files):
        print(f"{idx+1}. {file}")
    while True:
        choice = input("\nEnter the number of the query image to use: ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        else:
            print("Invalid choice. Try again.")

# main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5, help="number of results to return")
    parser.add_argument("--index_path", type=str, default=default_index)
    parser.add_argument("--metadata_path", type=str, default=default_metadata)
    parser.add_argument("--query_dir", type=str, default=default_query_dir)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    args = parser.parse_args()

    # prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # select query image
    query_image = select_query_image(args.query_dir)
    image_path = os.path.join(args.query_dir, query_image)

    # load model
    model, _ = clip.load("ViT-B/32", device=device)

    # embed image
    query_vec = embed_image(image_path, model)
    if query_vec is None:
        exit(1)

    # search and display results
    results = search(args.index_path, args.metadata_path, query_vec, args.top_k)
    print_results(image_path, results)

    # save results
    out_path = resolve_output_filename(args.output_dir, image_path)
    df = pd.DataFrame(results, columns=["url", "text"])
    df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")