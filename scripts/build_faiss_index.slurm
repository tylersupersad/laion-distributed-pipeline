#!/bin/bash
#SBATCH --job-name=faiss_build_array                           
#SBATCH --output=/home/almalinux/nfs/logs/slurm/faiss_%A_%a.out 
#SBATCH --error=/home/almalinux/nfs/logs/slurm/faiss_%A_%a.err  
#SBATCH --partition=batch                                      
#SBATCH --ntasks=1                                              
#SBATCH --cpus-per-task=4                                       
#SBATCH --mem=28G                                               
#SBATCH --time=04:00:00                                         

# define base paths for input, output, logs, and script
base="/home/almalinux/nfs"
input_dir="$base/outputs"                                       
output_dir="$base/index_shards"                                 
logs_dir="$base/logs"
faiss_script="$base/scripts/build_faiss_index.py"              

# create necessary log directories
mkdir -p "$logs_dir/faiss" "$logs_dir/slurm" "$output_dir"

# count how many parquet files are available for indexing
num_files=$(ls -1 "$input_dir"/clip_embeddings*.parquet 2>/dev/null | wc -l)

# skip task if slurm array index exceeds number of input files
if [ "$SLURM_ARRAY_TASK_ID" -ge "$num_files" ]; then
    echo "[SKIP] Task $SLURM_ARRAY_TASK_ID: No matching input file" >> "$logs_dir/faiss/faiss_skip.log"
    exit 0
fi

# launch the faiss indexing script with arguments
python3 "$faiss_script" \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" \
    --prefix "faiss_shard" \
    --normalize \
    --logs_dir "$logs_dir/faiss"
