import torch
import torch.nn as nn
import torchvision
import os


class ModelFactory:
    """Factory class for creating and configuring different model architectures"""
    
    def __init__(self, config):
        """Initialize the ModelFactory with configuration parameters
        Args:
            config (dict): Dictionary containing configuration parameters
        """
        self.model_type = config["model_type"]
        self.requires_grad = config.get("requires_grad", False)
        
    def create_model(self, num_classes, device):
        """Create and configure the model based on the specified type
        Args:
            num_classes (int): Number of output classes for the model
            device (torch.device): Device to move the model to (CPU or GPU)
        Returns:
            model (torch.nn.Module): Configured PyTorch model
        """
        
        if self.model_type == "shufflenet":
            model = self._create_shufflenet(num_classes)
        elif self.model_type == "mobilenet":
            model = self._create_mobilenet(num_classes)
        elif self.model_type == "efficientnet":
            model = self._create_efficientnet_v2_s(num_classes)
        elif self.model_type == "efficientnet_b0":
            model = self._create_efficientnet_b0(num_classes)
        elif self.model_type == "densenet":
            model = self._create_densenet(num_classes)
        elif self.model_type == "squeezenet":
            model = self._create_squeezenet(num_classes)
        elif self.model_type == "mnasnet":
            model = self._create_mnasnet(num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Supported types: shufflenet, mobilenet, efficientnet, "
                           f"efficientnet_b0, densenet, squeezenet, mnasnet")
        
        print(model)
        # Set initial gradient requirements
        for param in model.parameters():
            param.requires_grad = self.requires_grad
            
        # Move to device and handle multi-GPU
        print(f"Using {device} for training.")

        if torch.cuda.device_count() > 1 :
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            model = nn.DataParallel(model)
        elif torch.backends.mps.is_available() and torch.mps.device_count() > 1 :
            print(f"Using {torch.mps.device_count()} MPS for training.")
            model = nn.DataParallel(model)

        model = model.to(device)

        return model
    
    def _create_shufflenet(self, num_classes):
        """Create ShuffleNet V2 1.5x model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): ShuffleNet V2 1.5x model with modified classifier
        """
        model = torchvision.models.shufflenet_v2_x1_5(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    def _create_mobilenet(self, num_classes):
        """Create MobileNet V2 model
        Hint: Relaces ReLU6 with ReLU for better compatibility.
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): MobileNet V2 model with modified classifier
        """
        model = torchvision.models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        # Convert ReLU6 to ReLU for better compatibility
        model = self._convert_relu6_to_relu(model)
        return model
    
    def _create_efficientnet_v2_s(self, num_classes):
        """Create EfficientNet V2 Small model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): EfficientNet V2 Small model with modified classifier
        """
        model = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    def _create_efficientnet_b0(self, num_classes):
        """Create EfficientNet B0 model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): EfficientNet B0 model with modified classifier
        """
        model = torchvision.models.efficientnet_b0(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    def _create_densenet(self, num_classes):
        """Create DenseNet 121 model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): DenseNet 121 model with modified classifier
        """
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    
    def _create_squeezenet(self, num_classes):
        """Create SqueezeNet 1.1 model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): SqueezeNet 1.1 model with modified classifier
        """
        model = torchvision.models.squeezenet1_1(weights='DEFAULT')
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        return model
    
    def _create_mnasnet(self, num_classes):
        """Create MNASNet 0.5 model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): MNASNet 0.5 model with modified classifier
        """
        model = torchvision.models.mnasnet0_5(weights='DEFAULT')
        model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    def load_checkpoint(self, model, checkpoint_path):
        """Load model from checkpoint
        Args:
            model: The model to load the state into
            checkpoint_path (str): Path to the checkpoint file. If None, uses default path.
        Returns:
            bool: True if model was loaded successfully, False otherwise"""
        if os.path.exists(checkpoint_path):
            # check if pth or pt and handle accordingly
            if checkpoint_path.endswith('.pth'):
                model = torch.load(checkpoint_path)
            elif checkpoint_path.endswith('.pt'):
                model.load_state_dict(torch.load(checkpoint_path))
            print(f"Model loaded from {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            return False

    
    def _convert_relu6_to_relu(self, model):
        """Convert ReLU6 layers to ReLU layers in the model
        Args:
            model (torch.nn.Module): The model to convert
        Returns:
            model (torch.nn.Module): Model with ReLU6 layers replaced by ReLU
        """
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU6):
                inplace = child.inplace
                setattr(model, child_name, nn.ReLU(inplace=inplace))
            else:
                self._convert_relu6_to_relu(child)
        return model
    
    def get_model_info(self):
        """Get information about the model type
        Returns:
            str: Description of the model type
        """
        model_info = {
            "shufflenet": "ShuffleNet V2 1.5x - Lightweight CNN for mobile devices",
            "mobilenet": "MobileNet V2 - Efficient CNN for mobile and embedded devices",
            "efficientnet": "EfficientNet V2 Small - Balanced accuracy and efficiency",
            "efficientnet_b0": "EfficientNet B0 - Base EfficientNet model",
            "densenet": "DenseNet 121 - Dense connections for better feature reuse",
            "squeezenet": "SqueezeNet 1.1 - Very lightweight CNN",
            "mnasnet": "MNASNet 0.5 - Neural architecture search optimized model"
        }
        return model_info.get(self.model_type, "Unknown model type") 