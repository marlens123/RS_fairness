#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000

module load anaconda/3

conda activate rs_fairness

python -m src.compute_similarity --source_data "naip" --target_data "full"
python -m src.compute_similarity --source_data "naip" --target_data "rural"
python -m src.compute_similarity --source_data "naip" --target_data "urban"
python -m src.compute_similarity --source_data "sentinel2" --target_data "full"
python -m src.compute_similarity --source_data "sentinel2" --target_data "rural"
python -m src.compute_similarity --source_data "sentinel2" --target_data "urban"