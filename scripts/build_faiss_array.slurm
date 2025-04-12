#!/bin/bash
#SBATCH --job-name=faiss_build_array
#SBATCH --output=/home/almalinux/beegfs🐝/logs/faiss_%A_%a.out
#SBATCH --error=/home/almalinux/beegfs🐝/logs/faiss_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9999

# base paths
base=/home/almalinux/beegfs🐝
input_dir=$base/outputs
output_dir=$base/index_shards
logs_dir=$base/logs

# count clip embedding parquet files only
file_count=$(ls -1 "$input_dir"/clip_embeddings*.parquet 2>/dev/null | wc -l)

# exit if no files or task index too high
if [ "$file_count" -eq 0 ]; then
    echo "No embedding files found in $input_dir"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$file_count" ]; then
    echo "Skipping task $SLURM_ARRAY_TASK_ID: No matching file"
    exit 0
fi

# run faiss shard build
python3 build_faiss_index.py \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" \
    --prefix faiss_shard \
    --normalize
