import os
import io
import clip
import torch
import argparse
import pandas as pd
import pyarrow as pa
from PIL import Image
from tqdm import tqdm
import pyarrow.parquet as pq
from torchvision import transforms
from urllib.request import urlopen, Request

# get slurm array task id (used to select which parquet file to process)
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

# select device based on gpu availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# default transform (only used if clip-provided preprocess fails)
transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])

# load image from url and apply clip preprocess
def load_image(url, preprocess):
    try:
        # add user-agent header to avoid being blocked by servers
        headers = {'User-Agent': 'Mozilla/5.0'}
        with urlopen(Request(url, headers=headers), timeout=10) as r:
            img_data = r.read()
        # convert to rgb and apply transform
        return preprocess(Image.open(io.BytesIO(img_data)).convert("RGB"))
    except Exception:
        # return none if loading or transformation fails
        return None

# embed all valid images in a dataframe using clip
def embed_images(df, batch_size, model, preprocess):
    # store all final embeddings and their sample ids
    embeddings, ids = [], []              
    # temporary batch and its corresponding sample indices
    batch, batch_ids = [], []             

    # iterate through each row in the dataframe
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding task {task_id}"):
        # try to load image from url
        img = load_image(row["URL"], preprocess)
        if img is not None:
            batch.append(img)
            batch_ids.append(i)

        # run inference if batch is full or at the end of the dataframe
        if len(batch) >= batch_size or (i == len(df) - 1 and batch):
            try:
                # stack images into a batch tensor
                x = torch.stack(batch).to(device)  
                with torch.no_grad():
                    # compute embeddings on cpu
                    z = model.encode_image(x).cpu()  
                    
                # store all embeddings
                embeddings.extend(z)        
                # store corresponding sample ids
                ids.extend(batch_ids)       
            except Exception:
                # skip batch if inference fails
                pass
            
            # clear batch for next iteration
            batch, batch_ids = [], []   
            # release gpu memory    
            torch.cuda.empty_cache()        

    return embeddings, ids

# save the embeddings and metadata to a parquet file
def save_embeddings(embs, df, ids, output_dir, prefix):
    try:
        # build pyarrow table with sample ids, urls, texts, and vector embeddings
        table = pa.Table.from_pydict({
            "sample_id": [i for i in ids],
            "url": [df.loc[i, "URL"] for i in ids],
            "text": [df.loc[i, "TEXT"] for i in ids],
            "embedding": [e.numpy().tolist() for e in embs]
        })
        # write parquet file to output directory with task id suffix
        out_path = os.path.join(output_dir, f"{prefix}_task{task_id}.parquet")
        pq.write_table(table, out_path)
        
    except Exception as e:
        # print error if saving fails
        print(f"# error saving parquet for task {task_id}: {e}")

# main function: loads data, runs clip embedding, and saves results
def main(parquet_dir, output_dir, prefix, sample_count, batch_size):
    # get list of input parquet files and sort them
    files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])

    # check for valid slurm task id
    if task_id < 0 or task_id >= len(files):
        raise ValueError(f"# invalid slurm task id {task_id}")

    # read selected parquet file and limit to sample_count rows
    file_path = os.path.join(parquet_dir, files[task_id])
    df = pd.read_parquet(file_path).head(sample_count)

    # load clip model and preprocessing function
    model, preprocess = clip.load("ViT-B/32", device=device)

    # run embedding pipeline
    embs, ids = embed_images(df, batch_size, model, preprocess)

    # save outputs
    save_embeddings(embs, df, ids, output_dir, prefix)

# parse cli args and call main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path to input parquet files
    parser.add_argument("--parquet_dir", type=str, required=True)

    # directory to save output embeddings
    parser.add_argument("--output_dir", type=str, required=True)

    # filename prefix for output parquet
    parser.add_argument("--output_prefix", type=str, default="clip_embeddings")

    # number of samples to embed from each file
    parser.add_argument("--sample_count", type=int, default=1000)

    # batch size for clip inference
    parser.add_argument("--batch_size", type=int, default=64)

    # parse args and call main
    args = parser.parse_args()
    
    main(args.parquet_dir, args.output_dir, args.output_prefix, args.sample_count, args.batch_size)