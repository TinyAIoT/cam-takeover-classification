import os
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import Lambda as L
from torchvision.transforms import functional as F 
from torch.utils.data import DataLoader,ConcatDataset
from utils.training_utils import get_default_device, check_corrupted, to_device
from utils.transform_utils import SaveEvery_nth_image,CenterCrop,Image_Name_Saver,ImageFolderWithFilename


class DatasetHandler:
    def __init__(self, config):
        """Initialize the DatasetHandler with configuration parameters
        Args:
            config (dict): Dictionary containing configuration parameters
        """
        self.batch_size = config["batch_size"]
        #self.data_path = config["dataset_path"]
        self.num_workers = config["num_workers"]
        self.check_corrupted_images = config.get("check_corrupted_images", False)
        self.device = get_default_device()
        self.pin_memory = config.get("pin_memory", True) # Pin memory for faster data transfer to GPU, not sure if this is needed for MPS and CPU
        self.datasets = config.get("datasets", [])
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_names = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def build_transforms(self):
        transforms_built = []
        for dataset in self.datasets:
            dataset_transforms = []
            name_saver=Image_Name_Saver()
            dataset_transforms.append(name_saver)
            for t in dataset.get("transforms", []):
                
                if "resize" in t:
                    params = t["resize"]
                    height = params["height"]
                    width = params["width"]
                    dataset_transforms.append(transforms.Resize((height, width)))
                if "RandomCrop" in t:
                    params = t["RandomCrop"]
                    height = params["height"]
                    width = params["width"]
                    padding = params.get("padding", 0)
                    pad_if_needed = params.get("pad_if_needed", False)
                    fill = params.get("fill", 0)
                    padding_mode = params.get("padding_mode", "constant")
                    dataset_transforms.append(transforms.RandomCrop((height, width), padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode))
                if "rotate" in t:
                    degrees = t["rotate"].get("degrees", 0)
                    dataset_transforms.append(L(lambda img: F.rotate(img, degrees)))
                if "crop_relative" in t:
                    params = t["crop_relative"]
                    top = params.get("top", 0.0)
                    left = params.get("left", 0.0)
                    height = params["height"]
                    width = params["width"]
                    dataset_transforms.append(L(lambda img: F.resized_crop(img, int(top * img.height), int(left * img.width), int(height * img.height), int(width * img.width), (img.height, img.width))))
                if "CenterCrop" in t:
                    params = t["CenterCrop"]
                    side = params.get("side", "center")
                    height = params["height"]
                    width = params["width"]
                    dataset_transforms.append(CenterCrop(side,height, width))
                if "crop" in t:
                    params = t["crop"]
                    top = params.get("top", 0)
                    left = params.get("left", 0)
                    height = params["height"]
                    width = params["width"]
                    dataset_transforms.append(L(lambda img: F.crop(img, top, left, height, width)))
                if "save_img" in t:
                    params = t["save_img"]
                    n = params.get("n", 100)
                    save_dir = params.get("save_dir", "debug_images")
                    dataset_transforms.append(SaveEvery_nth_image(n=n, save_dir=save_dir,name_saver=name_saver))
                if "ToTensor" in t:
                    dataset_transforms.append(transforms.ToTensor())
                if "normalize" in t:
                    params = t["normalize"]
                    mean = params.get("mean", [0.485, 0.456, 0.406])
                    std = params.get("std", [0.229, 0.224, 0.225])
                    dataset_transforms.append(transforms.Normalize(mean=mean, std=std))
            transform = transforms.Compose(dataset_transforms)
            transforms_built.append((dataset["name"], transform))
        self.transforms_built = {name: transform for name, transform in transforms_built}
        print("Built the following transforms for datasets: \n")
        for name, transform in transforms_built:
            print(f"Dataset: {name}, Transform: {transform} \n")
                    
        return transform
    
    def load_data(self):
        '''
        Load (multiple) datasets and apply transformations
        '''
    
        train_paths = []
        val_paths = []
        test_paths = []
        self.build_transforms()
        for ds in self.datasets:
            train_path= os.path.join(ds["path"], "train")
            val_path = os.path.join(ds["path"], "val")
            test_path = os.path.join(ds["path"], "test")
            ds["train_path"] = train_path
            ds["val_path"] = val_path
            ds["test_path"] = test_path
            train_paths.append(train_path)
            val_paths.append(val_path)
            test_paths.append(test_path)
        
        
        for path in train_paths + val_paths + test_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset path {path} does not exist.")
        train_datasets = []
        val_datasets = []
        test_datasets = []
        for ds in self.datasets:
            name = ds["name"]
            train_path = ds["train_path"]
            val_path = ds["val_path"]
            test_path = ds["test_path"]
            transform = self.transforms_built[name]
            train_datasets.append(ImageFolderWithFilename(train_path, transform))
            val_datasets.append(ImageFolderWithFilename(val_path, transform))
            test_datasets.append(ImageFolderWithFilename(test_path, transform))
        self.train_data = ConcatDataset(train_datasets)
        self.val_data = ConcatDataset(val_datasets)
        self.test_data = ConcatDataset(test_datasets)
        if self.check_corrupted_images:
            print("Checking for corrupted images is not included yet for multiple datasets")
            # Check for corrupted images by trying to open and fully load each file
            for ds in self.train_data.datasets:
                check_corrupted(ds.samples, "train")
            for ds in self.val_data.datasets:
                check_corrupted(ds.samples, "val")
            for ds in self.test_data.datasets:
                check_corrupted(ds.samples, "test")
        
        all_classes = set()
        for ds in self.train_data.datasets:
            print(f"Classes in training dataset {ds.root}: \n {ds.class_to_idx} \n")
            all_classes.update(ds.classes)
            if all_classes != set(ds.classes):
                print(f"Warning: Datasets contain different classes. Check for correct index mapping")
            
        self.class_names = sorted(list(all_classes))
        print("All class names: \n")
        print(self.class_names)
        print("Train dataset size:", len(self.train_data))
        print("Validation dataset size:", len(self.val_data))
        print("Test dataset size:", len(self.test_data))
            
    def preprocess_data(self):
        """Create DataLoaders for training, validation, and testing"""
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)

    
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
        # Return the data loaders
        return self.train_loader, self.val_loader, self.test_loader
    
    def print_data_loader_structure(self, loader):
        """Return all data loaders
        Returns:
            tuple: Tuple containing train, validation, and test DataLoaders
        """
        # breakpoint()  # Debugging point to check if data loaders are created
        # print structure of tensor within data loaders
        if loader is None:
            raise ValueError("Data loader has not been created yet. Call preprocess_data() first.")
        for batch in loader:
            print(batch[0].shape, batch[1].shape)
            break
    
    def get_device(self):
        """Return the device being used
        Returns:
            torch.device: Device used for training (CPU, CUDA, or MPS)
        """
        return self.device 