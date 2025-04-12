#!/bin/bash
#SBATCH --job-name=merge_faiss
#SBATCH --output=/home/almalinux/beegfs🐝/logs/merge_faiss.out
#SBATCH --error=/home/almalinux/beegfs🐝/logs/merge_faiss.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# base paths
base=/home/almalinux/beegfs🐝
index_dir=$base/index_shards
output_dir=$base/faiss_index
output_index=$output_dir/merged.index
output_meta=$output_dir/merged_metadata.parquet

# create output dir if needed
mkdir -p "$output_dir"

# verify shard presence
shard_count=$(ls -1 "$index_dir"/*.index 2>/dev/null | wc -l)
if [ "$shard_count" -eq 0 ]; then
    echo "No index shards found in $index_dir"
    exit 1
fi

# run merging
python3 merge_faiss_shards.py \
    --index_dir "$index_dir" \
    --output_index "$output_index" \
    --output_metadata "$output_meta" \
    --normalize
