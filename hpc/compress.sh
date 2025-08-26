#!/bin/bash

##SBATCH --nodes=1
#SBATCH --ntasks=4                    # Total number of tasks
#SBATCH --gres=gpu:4                  # Request 2 GPUs
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=80GB                    # Memory per node

#SBATCH --partition=gpu3090,gpu2080,gpuhgx

#SBATCH --time=0-00:30:00

#SBATCH --job-name=quant_squeezy_org

#SBATCH --mail-type=ALL

#SBATCH --output /scratch/tmp/%u/cam-takeover-classification/tmp/compress/quantize_squeezy_org_%j.log

#load modules 
# module purge
# module load palma/2021a foss/2021a torchvision/0.11.1-CUDA-11.3.1  ONNX-Runtime/1.10.0-CUDA-11.3.1
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
pip uninstall -y torchvision
pip uninstall -y torch
pip uninstall -y pillow
pip install --user torch==2.2.0+cu121 torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# pip install --user torchvision==0.16.2
pip install --user tqdm
pip install --user pycocotools
pip install --user opencv-python
pip uninstall -y ppq esp-ppq
pip install --user esp-ppq


# pip list 
# module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
# module load TensorFlow/2.13.0
# module load scikit-learn/1.3.1
# sh ./install.sh
# . ./export.sh
# pip install onnxruntime pandas 
# # pip uninstall ppq
# pip install git+https://github.com/espressif/esp-ppq.git
# pip install onnx2pytorch
# pip install --user onnx2torch
# pip uninstall onnx2pytorch -y
# place of code in palma
home="$HOME"
wd="$WORK"/cam-takeover-classification
code="$HOME"/cam-takeover-classification


# python "$code"/compression/esp-dl/quantize_torch_model.py --working_dir $wd "$@"

log_path="$WORK"/cam-takeover-classification/tmp/compress/quantize_squeezy_org_"$SLURM_JOB_ID"
echo $log_path
# Define an array of input models as parameters
# opt_level=( 1 2 )
# iterations=( 2)
# value_threshold=( 0.2 0.5 0.8 1 1.5 2)
# including_bias=( False True )
# bias_multiplier=( 0.5 )
# including_act=( False True )
# act_multiplier=( 0.5 )

opt_level=( 2 )
iterations=( 3 )
value_threshold=( 0.8 )

python -c "import esp_ppq; print(dir(esp_ppq))"
python -c "import torchvision; print(dir(torchvision))"
python "$code"/compression/compressor.py --opt_level 2 --iterations 3 --value_threshold 0.8 --working_dir $wd

echo "end of compression for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
