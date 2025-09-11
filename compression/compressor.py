# Description: This script quantizes a pre-trained PyTorch model using the ESP-DL library.
# The script reads the configuration file and the model file, quantizes the model, and evaluates the quantized model.
# The script saves the quantized model and the evaluation results.
# The config file should contain the following fields:
# - batch_size: The batch size for the calibration and testing datasets.
# - input_model_path: The path to the pre-trained PyTorch model file.
# - dataset_path: The path to the dataset directory. The directory should contain the calibration and testing datasets.
# - output_path: The path to the output directory.
# - model_name: The name of the model.
# - quant_config: The quantization configuration. The supported values are None, "LayerwiseEqualization_quantization", and "MixedPrecision_quantization".

# -------------------------------------------
# Import Libraries
# --------------------------------------------
import os
from typing import Tuple, List, Tuple
import yaml
import json
import torch
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from utilities.calib_util import evaluate_ppq_module_with_pv, evaluate_torch_module_with_imagenet
from esp_ppq import QuantizationSettingFactory, QuantizationSetting
from esp_ppq.api import get_target_platform, espdl_quantize_torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import argparse
import torch.nn as nn
import pandas as pd
from utilities.dataloader_util import ImageFolderWithFilenameReturn

def get_default_device():
    """Pick GPU if available, else CPU
    Returns:
        torch.device: Device to use for training (CPU, CUDA, or MPS)
    """
    if torch.cuda.is_available():
        print(f"Using CUDA for training.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print(f"Using MPS for training.")
        return torch.device("mps")
    else:
        print(f"Using CPU for training.")
        return torch.device("cpu")

# -------------------------------------------
# Helper Functions
# --------------------------------------------

# Pretty print the quantization settings
def pretty_print_settings(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if hasattr(value, '__dict__'):
        if isinstance(value.__dict__, dict):      
          pretty_print_settings(value.__dict__, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

# Set the quantization settings for the model
def set_quant_settings(
    optim_quant_method: List[str] = None,
) -> QuantizationSetting:
    """Quantize onnx model with optim_quant_method.

    Args:
        optim_quant_method (List[str]): support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'
        -'MixedPrecision_quantization': if some layers in model have larger errors in 8-bit quantization, dispathching
                                        the layers to 16-bit quantization. You can remove or add layers according to your
                                        needs.
        -'LayerwiseEqualization_quantization'： using weight equalization strategy, which is proposed by Markus Nagel.
                                                Refer to paper https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf for more information.
                                                If ReLu6 layers are used in model, make sure to convert ReLU6 to ReLU for better precision.
    Returns:
        [tuple]: [QuantizationSetting, str]
    """
    quant_setting = QuantizationSettingFactory.espdl_setting()
    if optim_quant_method is not None and "None" not in optim_quant_method:
        if "MixedPrecision_quantization" in optim_quant_method:
            # These layers have larger errors in 8-bit quantization, dispatching to 16-bit quantization.
            # You can remove or add layers according to your needs.
            quant_setting.dispatching_table.append(
                "/features/features.1/conv/conv.0/conv.0.0/Conv",
                get_target_platform(TARGET, 16, True),
            )
            quant_setting.dispatching_table.append(
                "/features/features.16/conv/conv.1/conv.1.0/Conv",
                get_target_platform(TARGET, 16, True),
            )
        elif "LayerwiseEqualization_quantization" in optim_quant_method:
            # layerwise equalization
            quant_setting.equalization = True
            quant_setting.equalization_setting.iterations = args.iterations
            quant_setting.equalization_setting.value_threshold = args.value_threshold
            quant_setting.equalization_setting.opt_level = args.opt_level
            quant_setting.equalization_setting.including_bias = args.including_bias
            quant_setting.equalization_setting.bias_multiplier = args.bias_multiplier
            quant_setting.equalization_setting.including_act = args.including_act
            quant_setting.equalization_setting.act_multiplier = args.act_multiplier
            quant_setting.equalization_setting.interested_layers = None
            
            print("Quantization settings: ")
            pretty_print_settings(quant_setting.__dict__)
        else:
            raise ValueError(
                "Please set optim_quant_method correctly. Support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'"
            )
    return quant_setting


def collate_fn1(x: Tuple) -> torch.Tensor:
    return torch.cat([sample[0].unsqueeze(0) for sample in x], dim=0)


def collate_fn2(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="compression/configs/squeezenet.yaml", help="Path to the config file.")
    argparser.add_argument("--working_dir", type=str, default="./", help="Path to local data directory.")
    argparser.add_argument("--opt_level", type=int, default=2, help="Optimization level for equalization.")
    argparser.add_argument("--iterations", type=int, default=10, help="Number of iterations for equalization.")
    argparser.add_argument("--value_threshold", type=float, default=0.5, help="Value threshold for equalization.")
    argparser.add_argument("--including_bias", type=bool, default=False, help="Include bias in equalization.")
    argparser.add_argument("--bias_multiplier", type=float, default=0.5, help="Bias multiplier for equalization.")
    argparser.add_argument("--including_act", type=bool, default=False, help="Include activation in equalization.")
    argparser.add_argument("--act_multiplier", type=float, default=0.5, help="Activation multiplier for equalization.")
    args = argparser.parse_args()

    # Read the configuration file
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            input_model_subpath=config["input_model_path"] 
            dataset_subpath=config["dataset_path"] 
            output_subpath=config["output_path"] 
            input_model_path=os.path.join(args.working_dir,input_model_subpath)
            dataset_path=os.path.join(args.working_dir,dataset_subpath)
            output_path=os.path.join(args.working_dir,output_subpath)
            config["input_model_path"]=input_model_path
            config["dataset_path"]=dataset_path
            config["output_path"]=output_path
        print("Configuration:", json.dumps(config, sort_keys=True, indent=4))
    except Exception as e:
        # If the configuration file is not found or invalid, print an error message and exit
        print("Error reading the config file.")
        print(e)
        exit(1)

    # BATCH_SIZE = config["batch_size"]
    BATCH_SIZE = config["batch_size"]
    # DEVICE = "cpu"  #  'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
    DEVICE = "cpu" # Issues with inconsistent device types
    TARGET = "esp32s3"
    NUM_OF_BITS = 8 # 8-bit quantization
    TORCH_PATH = config["input_model_path"]    
    ESPDL_MODLE_PATH = config["output_path"] + config["model_name"] + "_" + str(args.opt_level) + "_" + str(args.iterations) + "_" + str(args.value_threshold) + ".espdl"
    CALIB_DIR = config["dataset_path"] + "train"
    TEST_DIR = config["dataset_path"] + "test"

    if "img_height" in config:
        IMAGE_HEIGHT = config["img_height"]
    else:
        IMAGE_HEIGHT = 24
    print("Image Height: ")
    print(IMAGE_HEIGHT)

    if "img_width" in config:
        IMAGE_WIDTH = config["img_width"]
    else:
        IMAGE_WIDTH = 24

    INPUT_SHAPE = [3, IMAGE_HEIGHT, IMAGE_WIDTH] # Torch format



    # -------------------------------------------
    # Prepare Calibration Dataset
    # --------------------------------------------
    if os.path.exists(CALIB_DIR):
        print(f"load imagenet calibration dataset from directory: {CALIB_DIR}")
        dataset = datasets.ImageFolder(
            CALIB_DIR,
            transforms.Compose(
                [
                    transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to [-1, 1]
                ]
            ),
        )
        
        # Fix: Only create subset up to the actual dataset size
        dataset_size = len(dataset)
        max_samples = min(1024, dataset_size)  # Use whichever is smaller
        print(f"Dataset size: {dataset_size}, using {max_samples} samples for calibration")
        
        if max_samples > 0:
            dataset = Subset(dataset, indices=[_ for _ in range(0, max_samples)])
        else:
            raise ValueError("Calibration dataset is empty")
            
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            collate_fn=collate_fn1,
        )
    else:
        raise ValueError(
                "Please provide valid calibration dataset path"
            )
        
    if os.path.exists(TEST_DIR):
        print(f"load imagenet testing dataset from directory: {TEST_DIR}")
        test_dataset = ImageFolderWithFilenameReturn(
            TEST_DIR,
            transforms.Compose(
                [
                    transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to [-1, 1]
                ]
            ),
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            drop_last=True,  # The onnx model does not support dynamic batchsize, and the data size of the last batch may not be aligned, so the last batch of data is discarded
        )
    else:
        raise ValueError(
                "Please provide valid testing dataset path"
            )

    # -------------------------------------------
    # Load Model
    # --------------------------------------------
    
    model = torch.load(TORCH_PATH)
    # torch_model = torch_model.to(DEVICE)
    
    # -------------------------------------------
    # Set Quantization Setting
    # --------------------------------------------
    
    quant_setting = set_quant_settings(
        config["quant_config"]
    )
    
    # -------------------------------------------
    # Evaluate Original Model
    # --------------------------------------------
    
    my_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    my_model.eval() # Set the model to evaluation mode
    
    # Freeze the model parameters
    for param in my_model.parameters():
        param.requires_grad = False
    
    # Save the model 
    # torch.save(my_model, config["output_path"] + config["model_name"] + "_" + str(args.opt_level) + "_" + str(args.iterations) + "_" + str(args.value_threshold) + ".pth")

    # Evaluate the model
    test = evaluate_torch_module_with_imagenet(
        model=model,
        batchsize= BATCH_SIZE,
        device=get_default_device(),
        imagenet_validation_loader=test_dataloader,
        verbose=True,
        img_height=IMAGE_HEIGHT,
        img_width=IMAGE_WIDTH,
        print_confusion_matrix=False,
        confusion_matrix_path=config["output_path"] + config["model_name"] + "_before_quantization_confusion_matrix.png",
    )

    # -------------------------------------------
    # Quantize Model
    # --------------------------------------------
    
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODLE_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=[1] + INPUT_SHAPE,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn2,
        setting=quant_setting,
        device="cuda:0",
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=1,
    )

    # -------------------------------------------
    # Evaluate Quantized Model.
    # --------------------------------------------

    quant_test = evaluate_ppq_module_with_pv(
        model=quant_ppq_graph,
        imagenet_validation_loader=test_dataloader,
        batchsize=BATCH_SIZE,
        device="cuda:0",
        verbose=1,
        print_confusion_matrix=True,
        img_height=IMAGE_HEIGHT,
        img_width=IMAGE_WIDTH,
        confusion_matrix_path=config["output_path"] + config["model_name"] + "_" + str(args.opt_level) + "_" + str(args.iterations) + "_" + str(args.value_threshold) + "_confusion_matrix.png",
        mosaic_path=config["output_path"] + config["model_name"] + "_" + str(args.opt_level) + "_" + str(args.iterations) + "_" + str(args.value_threshold) + "_mosaic_pred.jpg"
    )

    top1_test=sum(test["top1_accuracy"]) / len(test["top1_accuracy"])
    top5_test=sum(test["top5_accuracy"]) / len(test["top5_accuracy"])
    top1_quant=sum(quant_test["top1_accuracy"]) / len(quant_test["top1_accuracy"])
    top5_quant=sum(quant_test["top5_accuracy"]) / len(quant_test["top5_accuracy"])

    # concatente the results and export to a csv file
    results = {
        "top1_test": [top1_test],
        "top5_test": [top5_test],
        "top1_test_quant": [top1_quant],
        "top5_test_quant": [top5_quant],
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(config["output_path"] + config["model_name"] + "_quant-metrics.csv", index=False)