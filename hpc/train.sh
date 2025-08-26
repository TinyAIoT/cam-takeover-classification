#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # Total number of tasks
#SBATCH --gres=gpu:1                  # Request 4 GPUs
#SBATCH --cpus-per-task=16            # Number of CPU cores
# try --mem=16GB                    # Memory per node
#SBATCH --mem-per-cpu=2GB
#SBATCH --partition=requeue-gpu
# Use one of these options:
#   private gpus:                 requeue-gpu
#   all (non-private) zen3 gpus:      gpuv100,gpu2080,gpua100,gputitanrtx,gpu3090,gpuhgx
#   express gpu:                      gpuexpress


#SBATCH --time=0-02:00:00

#SBATCH --job-name=train_squeezy_org

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/cam-takeover-classification/tmp/training/train_squeezy_org_%j.log

#load modules 
module purge
# PyTorch/2.1.2-CUDA-12.1.1 
module load palma/2023a foss/2023a scikit-learn/1.3.1 matplotlib/3.7.2 
# module load palma/2021a foss/2021a torchvision/0.11.1-CUDA-11.3.1  ONNX-Runtime/1.10.0-CUDA-11.3.1 scikit-learn/0.24.2 matplotlib/3.4.2
# pip install --user matplotlib
pip install --user pyyaml
pip uninstall torch torchvision torchaudio -y
pip cache purge
# Remove any leftover .pyc or .so files if needed:
find ~/.local/lib/python3.11/site-packages/ -name "*torch*" -exec rm -rf {} +
find ~/.local/lib/python3.11/site-packages/ -name "*vision*" -exec rm -rf {} +
pip uninstall --user -y torchvision
pip uninstall --user -y torch
pip uninstall --user -y pillow ppq
pip install --user torch==2.2.0+cu121 torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# pip install --user torchvision==0.16.2
pip install --user tqdm
pip install --user pycocotools
pip install --user opencv-python
# For debugging use: 
# pip install torchprofile 
# place of code in palma
wd="$WORK"/cam-takeover-classification
code="$HOME"/cam-takeover-classification

# run code with flags
# run sbatch with flag --config path/to/config.yaml
python "$code"/training/run_transfer_learning_pipeline.py --working_dir $wd "$@"
echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`