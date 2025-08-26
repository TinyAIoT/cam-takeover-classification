import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.training_utils import get_default_device, check_corrupted, to_device, BottomCenterCrop

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        """Number of batches
        Returns:
            int: number of batches in the DataLoader"""
        return len(self.dl)


class DatasetHandler:
    def __init__(self, config):
        """Initialize the DatasetHandler with configuration parameters
        Args:
            config (dict): Dictionary containing configuration parameters
        """
        self.img_height = config["img_height"]
        self.img_width = config["img_width"]
        self.batch_size = config["batch_size"]
        self.data_path = config["dataset_path"]
        self.num_workers = config["num_workers"]
        self.check_corrupted_images = config.get("check_corrupted_images", False)
        self.device = get_default_device()
        self.pin_memory = config.get("pin_memory", True) # Pin memory for faster data transfer to GPU, not sure if this is needed for MPS and CPU
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_names = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def load_data(self):
        """Load datasets from the specified data path"""
        transform = transforms.Compose([
            BottomCenterCrop(self.img_height, self.img_width),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_data = datasets.ImageFolder(os.path.join(self.data_path, "train"), transform)
        self.val_data = datasets.ImageFolder(os.path.join(self.data_path, "val"), transform)
        self.test_data = datasets.ImageFolder(os.path.join(self.data_path, "test"), transform)

        if self.check_corrupted_images:
            print("Checking for corrupted images in the dataset...")
            # Check for corrupted images by trying to open and fully load each file
            check_corrupted(self.train_data.samples, "train")
            check_corrupted(self.val_data.samples, "val")
            check_corrupted(self.test_data.samples, "test")

        self.class_names = self.train_data.classes
        print("Class names: ")
        print(self.class_names)
        print("Train dataset size:", len(self.train_data))
        print("Validation dataset size:", len(self.val_data))
        print("Test dataset size:", len(self.test_data))
    
    def preprocess_data(self):
        """Create DataLoaders for training, validation, and testing"""
        self.train_loader = DeviceDataLoader(
            DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory), 
            self.device
        )
        self.val_loader = DeviceDataLoader(
            DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory), 
            self.device
        )
        self.test_loader = DeviceDataLoader(
            DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory), 
            self.device
        )
    
    def get_class_names(self):
        """Return the class names
        Returns:
            list: List of class names in the dataset
        """
        return self.class_names
    
    def get_data_loaders(self):
        """Return all data loaders
        Returns:
            tuple: Tuple containing train, validation, and test DataLoaders
        """
        # breakpoint()  # Debugging point to check if data loaders are created
        # print structure of tensor within data loaders
        if self.train_loader is None or self.val_loader is None or self.test_loader is None:
            raise ValueError("Data loaders have not been created. Call preprocess_data() first.")
        print("Train loader structure:")
        for batch in self.train_loader:
            print(batch[0].shape, batch[1].shape)
            break
        print("Validation loader structure:")
        for batch in self.val_loader:
            print(batch[0].shape, batch[1].shape)
            break
        print("Test loader structure:")
        for batch in self.test_loader:
            print(batch[0].shape, batch[1].shape)
            break
        # Return the data loaders
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_device(self):
        """Return the device being used
        Returns:
            torch.device: Device used for training (CPU, CUDA, or MPS)
        """
        return self.device 