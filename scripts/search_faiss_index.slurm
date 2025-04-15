#!/bin/bash
#SBATCH --job-name=search_faiss                                   
#SBATCH --output=/home/almalinux/nfs/logs/slurm/search_faiss_%j.out  
#SBATCH --error=/home/almalinux/nfs/logs/slurm/search_faiss_%j.err   
#SBATCH --ntasks=1                                                
#SBATCH --cpus-per-task=2                                           
#SBATCH --mem=8G                                                   
#SBATCH --time=0:30:00                                             

# get image name passed as the first command-line argument
image="$1"

# run the faiss image search script with top_k = 5
python3 /home/almalinux/nfs/scripts/search_faiss_index.py "$image" --top_k 5
