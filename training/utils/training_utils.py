import torch
from torchvision import datasets, transforms
from PIL import Image, UnidentifiedImageError

def check_corrupted(samples, split_name):
    """Check for corrupted images in the dataset by trying to open and load each file
    Args:
        samples (list): List of tuples containing image paths and labels
        split_name (str): Name of the dataset split (e.g., "train", "val", "test")
    Returns:
        list: List of paths to corrupted images
    """
    corrupted = []
    for path, _ in samples:
        try:
            # First pass: verify header
            with Image.open(path) as img:
                img.verify()
            # Second pass: force actual decoding
            with Image.open(path) as img:
                img.load()
        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"Corrupted image in {split_name}: {path} ({e})")
            corrupted.append(path)
    if not corrupted:
        print(f"No corrupted images found in {split_name}.")
    else:
        print(f"Found {len(corrupted)} corrupted images in {split_name}.")
    return corrupted

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

def to_device(data, device):
    """Move tensor(s) to chosen device
    Args:
        data (tensor or list/tuple of tensors): Data to move to device
        device (torch.device): Device to move the data to (CPU, CUDA, or MPS)
    Returns:
        tensor or list/tuple of tensors: Data moved to the specified device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
