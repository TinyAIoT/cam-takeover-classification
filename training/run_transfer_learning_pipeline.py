#!/usr/bin/env python3
"""
Example usage of the improved transfer learning structure (V2).

This script demonstrates the new six-component architecture that properly handles:
1. DatasetHandler - Dataset loading and preprocessing
2. ModelFactory - Model creation and architecture selection
3. LearningConfigurator - Handles transfer learning and fine-tuning (freeze backbone, train head, unfreeze layers, small LR)
5. Trainer - Generic training loop (used by both phases)
6. Evaluator - Model evaluation and visualization
"""
# Imports
import yaml
import argparse
import os
import json
from orchestrator import Orchestrator

def run_full_pipeline(config):
    print("\n" + "=" * 70)
    print("TYPICAL TRANSFER LEARNING WORKFLOW")
    print("=" * 70)

    orchestrator = Orchestrator(config)
    
    print("\nWorkflow Steps:")
    print("1. Load pretrained model (e.g., MobileNet trained on ImageNet)")
    print("2. Freeze all backbone layers")
    print("3. Replace classification head with new head for your classes")
    print("4. Train only the classification head (Transfer Learning)")
    print("5. Optionally unfreeze some backbone layers")
    print("6. Train with smaller learning rate (Fine-tuning)")
    print("7. Evaluate final model")
    
    orchestrator.print_component_info()

    # Run the workflow
    results = orchestrator.run_transfer_learning_pipeline()
    # results = orchestrator.run_transfer_learning_only()
    # results = orchestrator.run_fine_tuning_only()
    
    print(f"\nFinal Results:")
    if 'transfer_learning_history' in results:
        print(f"Transfer Learning Best Accuracy: {max(results['transfer_learning_history']['val_accuracy']):.4f}")
    if 'fine_tuning_history' in results:
        print(f"Fine-tuning Best Accuracy: {max(results['fine_tuning_history']['val_accuracy']):.4f}")
    if 'evaluation_results' in results:
        print(f"Final Test Accuracy: {results['evaluation_results']['test_accuracy']:.4f}")
    
    
    
def compare_different_strategies(config, fine_tune_configs=None):
    print("\n=== Comparing Different Approaches ===")
    
    for ft_config in fine_tune_configs:
        print(f"\n--- Testing {ft_config['name']} ---")
        test_config = config.copy()
        test_config.update(ft_config)
        test_config["model_name"] = f"example_{ft_config['name']}"
        
        custom_orchestrator = Orchestrator(test_config)
        custom_orchestrator.print_component_info()
        
        results = custom_orchestrator.run_transfer_learning_pipeline()

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    parser.add_argument("--working_dir", type=str, default="./", help="Path to local data directory.")
    args=parser.parse_args()

    # Read the configuration file
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            #dataset_subpath=config["dataset_path"]
            
            output_subpath=config["output_path"] 
            dataset_subpaths=[ds["path"] for ds in config["datasets"]]
            for ds in config["datasets"]:
                ds["path"]=os.path.join(args.working_dir,ds["path"])
            output_path=os.path.join(args.working_dir,output_subpath)
            config["output_path"]=output_path
            
        
        # Set default parameters if missing
        default_params = {
            "img_height": 224,
            "img_width": 224,
            "batch_size": 32,
            "patience": 5,
            
            "model_name": "example_transfer_learning",
            "model_type": "squeezenet",
            "output_path": "./outputs",
            "num_workers": 4,
            "transfer_learning_epochs": 10,
            "transfer_learning_rate": 0.001,
            "transfer_weight_decay": 0.0001,
            "fine_tune": False,
            "fine_tune_epochs": 10,
            "fine_tuning_learning_rate": 0.0001,
            "fine_tuning_weight_decay": 0.0001,
            "num_unfreezed_feature_layers": 2,
            "requires_grad": False,
            "check_corrupted_images": False,
            "pin_memory": True,
            "profile": False
        }
        for key, default_value in default_params.items():
            config.setdefault(key, default_value)    
            
            
        print("Configuration:", json.dumps(config, sort_keys=True, indent=4))
    except Exception as e:
        # If the configuration file is not found or invalid, print an error message and exit
        print("Error reading the config file.")
        print(e)
        exit(1)
        
        
    # Run the full transfer learning + fine-tuning pipeline
    run_full_pipeline(config)
    
    
    # # Use the following code to compare across different fine tuning strategies:
    # 
    # fine_tune_configs = [
    #     {"num_unfreezed_feature_layers": 0, "name": "transfer_learning_only"},
    #     {"num_unfreezed_feature_layers": 1, "name": "fine_tune_1_layer"},
    #     {"num_unfreezed_feature_layers": 2, "name": "fine_tune_2_layers"},
    #     {"num_unfreezed_feature_layers": 3, "name": "fine_tune_3_layers"},
    # ]
    # 
    # compare_different_strategies(config, fine_tune_configs)
    
    
    
    
    