#!/bin/bash

##SBATCH --nodes=1
#SBATCH --ntasks=1                    # Total number of tasks
#SBATCH --gres=gpu:1                  # Request 2 GPUs
#SBATCH --cpus-per-task=2             # Number of CPU cores
#SBATCH --mem=10GB                    # Memory per node

#SBATCH --partition=gpu2080,gpua100,gputitanrtx,gpu3090,gpuhgx
# gpu3090,gpu2080,gpuhgx

#SBATCH --time=0-00:30:00

#SBATCH --job-name=cleanup_env

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/cam-takeover-classification/tmp/cleanup_%j.log

#load modules 
module purge

pip uninstall esp-ppq -y
pip uninstall ppq -y
pip uninstall pyyaml -y
pip uninstall onnxruntime -y
pip uninstall pandas -y
pip uninstall onnx2torch -y
pip uninstall onnx2pytorch -y

module load palma/2023a foss/2023a Seaborn/0.13.2 PyTorch/2.1.2-CUDA-12.1.1 scikit-learn/1.3.1 matplotlib/3.7.2

pip uninstall esp-ppq -y
pip uninstall ppq -y
pip uninstall pyyaml -y
pip uninstall onnxruntime -y
pip uninstall pandas -y
pip uninstall onnx2torch -y
pip uninstall onnx2pytorch -y

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`