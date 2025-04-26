#!/bin/bash
#SBATCH --job-name=clip_embed_array                      
#SBATCH --output=/home/almalinux/nfs/logs/slurm/embed_%A_%a.out  
#SBATCH --error=/home/almalinux/nfs/logs/slurm/embed_%A_%a.err   
#SBATCH --partition=batch                                
#SBATCH --ntasks=1                                      
#SBATCH --cpus-per-task=4                          
#SBATCH --mem=28G                                     
#SBATCH --time=30:00:00                                   

# define base paths for data, logs, and scripts
base="/home/almalinux/nfs"
input_dir="$base/laion"
output_dir="$base/outputs"
logs_dir="$base/logs"
embed_script="$base/scripts/embed_clip.py"

# create necessary directories if not exist
mkdir -p "$logs_dir/embed" "$logs_dir/slurm" "$output_dir"

# count number of available parquet files
num_files=$(ls -1 "$input_dir"/*.parquet 2>/dev/null | wc -l)

# skip task if slurm array index exceeds number of files
if [ "$SLURM_ARRAY_TASK_ID" -ge "$num_files" ]; then
    echo "[SKIP] Task $SLURM_ARRAY_TASK_ID: No matching input file" >> "$logs_dir/embed_skip.log"
    exit 0
fi

# run the embedding script with arguments
python3 "$embed_script" \
    --parquet_dir "$input_dir" \
    --output_dir "$output_dir" \
    --output_prefix "clip_embeddings" \
    --sample_count 16000 \
    --batch_size 64
