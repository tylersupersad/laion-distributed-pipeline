#!/bin/bash
#SBATCH --job-name=clip_embed_array
#SBATCH --output=/home/almalinux/beegfs🐝/logs/embed_%A_%a.out
#SBATCH --error=/home/almalinux/beegfs🐝/logs/embed_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9999

# base paths
base=/home/almalinux/beegfs🐝
input_dir=$base/laion
output_dir=$base/outputs
logs_dir=$base/logs

# get number of files
num_files=$(ls -1 "$input_dir"/*.parquet | wc -l)

# skip task if beyond file count
if [ "$SLURM_ARRAY_TASK_ID" -ge "$num_files" ]; then
    echo "Skipping task $SLURM_ARRAY_TASK_ID: No matching file"
    exit 0
fi

# run embedding
python3 embed_clip.py \
    --parquet_dir "$input_dir" \
    --output_dir "$output_dir" \
    --output_prefix clip_embeddings \
    --sample_count 100000 \
    --batch_size 64