#!/bin/bash
#SBATCH --job-name=search_faiss_array
#SBATCH --output=/home/almalinux/beegfs🐝/logs/search_%A_%a.out
#SBATCH --error=/home/almalinux/beegfs🐝/logs/search_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

# define base paths
base=/home/almalinux/beegfs🐝
index_path=$base/faiss_index/merged.index
metadata_path=$base/faiss_index/merged_metadata.parquet
query_glob="$base/query_images/*.jpg"

# verify query images exist
query_count=$(ls $query_glob 2>/dev/null | wc -l)
if [ "$query_count" -eq 0 ]; then
    echo "No query images found in $query_glob"
    exit 1
fi

# run the search
srun python3 search_faiss_index.py \
    --index_path "$index_path" \
    --metadata_path "$metadata_path" \
    --query_glob "$query_glob" \
    --top_k 5 \
    --normalize