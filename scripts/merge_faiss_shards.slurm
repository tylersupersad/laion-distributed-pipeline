#!/bin/bash
#SBATCH --job-name=merge_faiss                                     
#SBATCH --output=/home/almalinux/nfs/logs/slurm/merge_faiss.out   
#SBATCH --error=/home/almalinux/nfs/logs/slurm/merge_faiss.err     
#SBATCH --ntasks=1                                                  
#SBATCH --cpus-per-task=4                                           
#SBATCH --mem=28G                                                  
#SBATCH --time=02:00:00                                              

# define base paths
base=/home/almalinux/nfs
index_dir=$base/index_shards                                       
output_dir=$base/faiss_index                                       
log_dir=$base/logs/merge                                           
output_index=$output_dir/merged.index                             
output_meta=$output_dir/merged_metadata.parquet                    

# create output and log directories if they don't exist
mkdir -p "$output_dir" "$log_dir"

# check if there are any index shards to merge
shard_count=$(ls -1 "$index_dir"/*.index 2>/dev/null | wc -l)
if [ "$shard_count" -eq 0 ]; then
    echo "No index shards found in $index_dir" >> "$log_dir/merge_faiss.err"
    exit 1
fi

# run the faiss merging script with normalization enabled
python3 merge_faiss_shards.py \
    --index_dir "$index_dir" \
    --output_index "$output_index" \
    --output_metadata "$output_meta" \
    --normalize \
    --log_dir "$log_dir"
