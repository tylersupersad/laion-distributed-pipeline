#!/usr/bin/env python3

import os
import io
import clip
import torch
import logging
import argparse
import pandas as pd
import pyarrow as pa
from PIL import Image
from tqdm import tqdm
import pyarrow.parquet as pq
from torchvision import transforms
from urllib.request import urlopen, Request

# get slurm array task id (used to select the specific parquet file)
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

# configure per-task logging to track progress and errors
log_path = f"/home/almalinux/nfs/logs/embed/embed_task{task_id}.log"
logging.basicConfig(
    filename=log_path,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()

# detect gpu if available, fallback to cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# define a manual transform in case clip's default fails
transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])

# load and preprocess image from url with retry mechanism
def load_image(url, preprocess, retries=2):
    headers = {'User-Agent': 'Mozilla/5.0'}
    for _ in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=10) as r:
                img_data = r.read()
            return preprocess(Image.open(io.BytesIO(img_data)).convert("RGB"))
        except Exception:
            continue
    return None

# embed all images in the dataframe using clip model
def embed_images(df, batch_size, model, preprocess):
    embeddings, ids = [], []
    batch, batch_ids = [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding task {task_id}"):
        img = load_image(row["URL"], preprocess)
        if img is not None:
            batch.append(img)
            batch_ids.append(i)

        # when batch is ready or end of data, run embedding
        if len(batch) >= batch_size or (i == len(df) - 1 and batch):
            log.info(f"Embedding batch of {len(batch)} images (task {task_id})")
            try:
                x = torch.stack(batch).to(device)
                with torch.no_grad():
                    z = model.encode_image(x).cpu()
                embeddings.extend(z)
                ids.extend(batch_ids)
                log.info(f"Successfully embedded batch of {len(batch)} images")
            except Exception as e:
                log.warning(f"Error embedding batch ending at row {i}: {e}")
            batch, batch_ids = [], []
            torch.cuda.empty_cache()

    return embeddings, ids

# save embeddings and metadata to parquet format
def save_embeddings(embs, df, ids, output_dir, prefix):
    try:
        table = pa.Table.from_pydict({
            "sample_id": [i for i in ids],
            "url": [df.loc[i, "URL"] for i in ids],
            "text": [df.loc[i, "TEXT"] for i in ids],
            "embedding": [e.numpy().tolist() for e in embs]
        })
        out_path = os.path.join(output_dir, f"{prefix}_task{task_id}.parquet")
        pq.write_table(table, out_path)
        log.info(f"💾 saved {len(embs)} embeddings to {out_path}")
    except Exception as e:
        log.error(f"Failed to save parquet file: {e}")

# main function to coordinate loading, embedding, and saving
def main(parquet_dir, output_dir, prefix, sample_count, batch_size):
    # collect all available parquet files
    files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])

    # ensure task id is valid
    if task_id < 0 or task_id >= len(files):
        raise ValueError(f"Invalid slurm task id: {task_id}")

    # select parquet file assigned to this task
    file_path = os.path.join(parquet_dir, files[task_id])
    log.info(f"Task {task_id} started: Processing file {file_path}")
    log.info(f"Sampling up to {sample_count} rows | batch size = {batch_size}")

    # load and sample the dataframe
    df = pd.read_parquet(file_path).head(sample_count)

    # load clip model and preprocessing pipeline
    model, preprocess = clip.load("ViT-B/32", device=device)
    log.info(f"Loaded CLIP model ViT-B/32 on device: {device}")

    # run embedding for all sampled images
    embs, ids = embed_images(df, batch_size, model, preprocess)

    # save final outputs to parquet
    save_embeddings(embs, df, ids, output_dir, prefix)

    log.info(f"Task {task_id} completed: {len(embs)} images embedded")

# entry point for slurm array job
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # directory containing input parquet files
    parser.add_argument("--parquet_dir", type=str, required=True)     
    
    # where to save output parquet embeddings
    parser.add_argument("--output_dir", type=str, required=True)   
        
    # output file prefix
    parser.add_argument("--output_prefix", type=str, default="clip_embeddings")  
    
    # how many rows to sample from each parquet
    parser.add_argument("--sample_count", type=int, default=1000)    
      
    # image embedding batch size
    parser.add_argument("--batch_size", type=int, default=64)   
           
    args = parser.parse_args()

    main(args.parquet_dir, args.output_dir, args.output_prefix, args.sample_count, args.batch_size)