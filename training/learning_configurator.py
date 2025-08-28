import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.profiler
from tqdm import tqdm


class LearningConfigurator:
    """Class for handling transfer learning and fine-tuning of models"""
    
    def __init__(self):
        """No initialization needed. Just a placeholder for future extensions."""
        pass
                
    def prepare_model_for_transfer_learning(self, model):
        """Prepare model for transfer learning by freezing backbone layers
        Args:
            model (torch.nn.Module): The model to prepare for transfer learning.
        Returns:
            torch.nn.Module: The modified model with frozen backbone layers and trainable classification layer.
        """
        print("Preparing model for transfer learning...")
        
        # Freeze all backbone layers
        self._freeze_backbone_layers(model)
        
        # Make classification layer trainable
        self._make_classification_layer_trainable(model)
        
        # Print trainable parameters info
        trainable_params, total_params = self._get_trainable_parameters_count(model)
        print(f"Transfer learning - Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        # Print trainable layers
        self.print_trainable_layers(model)
        
        self.get_trainable_parameters_info(model)
        
        return model
    
    def prepare_model_for_fine_tuning(self, model, num_unfreezed_feature_layers):
        """Prepare model for fine-tuning by unfreezing selected layers
        Args:
            model (torch.nn.Module): The model to prepare for fine-tuning.
            num_unfreezed_feature_layers (int): Number of feature layers to unfreeze for fine-tuning.
        Returns:
            torch.nn.Module: The modified model with specified layers unfrozen and classification layer trainable.
        """
        print("Preparing model for fine-tuning...")\
        
        # Freeze all backbone layers 
        self._freeze_backbone_layers(model)

        # Unfreeze selected feature layers
        if num_unfreezed_feature_layers > 0:
            self._unfreeze_last_x_feature_layers(model, num_unfreezed_feature_layers)
        
        # Keep classification layer trainable
        self._make_classification_layer_trainable(model)
        
        # Print trainable parameters info
        trainable_params, total_params = self._get_trainable_parameters_count(model)
        print(f"Fine-tuning - Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        # Print trainable layers
        self.print_trainable_layers(model)
        
        self.get_trainable_parameters_info(model)
        
        return model
    
    def _freeze_backbone_layers(self, model):
        """Freeze all backbone layers (everything except classification layer)
        Args:
            model (torch.nn.Module): The model whose backbone layers to freeze.
        """
        if hasattr(model, 'module'):
            # Handle DataParallel models
            for name, param in model.module.named_parameters():
                if not name.startswith('classifier') and not name.startswith('fc'):
                    param.requires_grad = False
        else:
            # Handle regular models
            for name, param in model.named_parameters():
                if not name.startswith('classifier') and not name.startswith('fc'):
                    param.requires_grad = False
        print("Backbone layers frozen")
    
    
    def _make_classification_layer_trainable(self, model):
        """Make classification layer trainable
        Args:
            model (torch.nn.Module): The model whose classification layer to make trainable.
        """
        if hasattr(model, 'module'):
            # Handle DataParallel models
            for name, param in model.module.named_parameters():
                if name.startswith('classifier') or name.startswith('fc'):
                    param.requires_grad = True
        else:
            # Handle regular models
            for name, param in model.named_parameters():
                if name.startswith('classifier') or name.startswith('fc'):
                    param.requires_grad = True
        print("Classification layer made trainable")
        
    
    def _unfreeze_last_x_feature_layers(self, model, num_unfreezed_feature_layers):
        """Unfreeze the last x feature layers for fine-tuning
        Args:
            model (torch.nn.Module): The model whose feature layers to unfreeze.
            num_unfreezed_feature_layers (int): Number of feature layers to unfreeze.
        """
        x = num_unfreezed_feature_layers
        
        if hasattr(model, 'module'):
            # Handle DataParallel models
            if hasattr(model.module, 'features'):
                for i in range(len(model.module.features) - x, len(model.module.features)):
                    for param in model.module.features[i].parameters():
                        param.requires_grad = True
        else:
            # Handle regular models
            if hasattr(model, 'features'):
                for i in range(len(model.features) - x, len(model.features)):
                    for param in model.features[i].parameters():
                        param.requires_grad = True
        
        print(f"Last {x} feature layers unfrozen for fine-tuning")
    
    def _get_trainable_parameters_count(self, model):
        """Get the count of trainable parameters
        Args:
            model (torch.nn.Module): The model to analyze.
        Returns:
            tuple: (trainable_params, total_params) - Number of trainable parameters and total parameters in the model.
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params
    
    def print_trainable_layers(self, model):
        """Print which layers are trainable
        Args:
            model (torch.nn.Module): The model to analyze.
        """
        print("\nTrainable layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  ✓ {name}")
        print("\nFrozen layers:")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"  ✗ {name}")
    
    
    def get_trainable_parameters_info(self, model):
        """Get detailed information about trainable parameters
        Args:
            model (torch.nn.Module): The model to analyze.
        Returns:
            dict: A dictionary containing total trainable parameters, total parameters, 
                  percentage of trainable parameters, and breakdown by layer type.
        """
        trainable_params, total_params = self._get_trainable_parameters_count(model)
        
        # Count parameters by layer type
        feature_params = 0
        classifier_params = 0
        other_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'features' in name:
                    feature_params += param.numel()
                elif 'classifier' in name or 'fc' in name:
                    classifier_params += param.numel()
                else:
                    other_params += param.numel()
        
        info = {
            'total_trainable': trainable_params,
            'total_params': total_params,
            'trainable_percentage': trainable_params / total_params * 100,
            'feature_params': feature_params,
            'classifier_params': classifier_params,
            'other_params': other_params
        }
        
        return info 