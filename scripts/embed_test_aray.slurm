#!/bin/bash
#SBATCH --job-name=clip_embed_test
#SBATCH --output=/home/almalinux/beegfs🐝/logs/test_embed_%A_%a.out
#SBATCH --error=/home/almalinux/beegfs🐝/logs/test_embed_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-99

# base directories
base=/home/almalinux/beegfs🐝
input_dir=$base/test_inputs
output_dir=$base/test_outputs
logs_dir=$base/logs

# get all test files
parquet_files=($(ls "$input_dir"/*.parquet 2>/dev/null))
file_count=${#parquet_files[@]}

# check for valid task
if [ "$SLURM_ARRAY_TASK_ID" -ge "$file_count" ]; then
    echo "Skipping task $SLURM_ARRAY_TASK_ID: No matching file"
    exit 0
fi

# select file to process
parquet_file=${parquet_files[$SLURM_ARRAY_TASK_ID]}
filename=$(basename "$parquet_file" .parquet)

# run embedding
python3 embed_clip.py \
    --parquet_dir "$input_dir" \
    --output_dir "$output_dir" \
    --output_prefix test_embedding \
    --sample_count 500 \
    --batch_size 32