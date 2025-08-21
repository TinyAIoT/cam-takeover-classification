import os
import shutil
import random
from pathlib import Path

def split_data():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    classified_dir = Path("data/classified")
    split_dir = Path("data/split_classified")
    
    # Get all class directories (subfolders in classified_dir)
    class_dirs = [d for d in classified_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
    
    if not class_dirs:
        print(f"No class directories found in {classified_dir}")
        return
    
    # Create output directory structure
    for split in ['train', 'test', 'val']:
        for class_dir in class_dirs:
            class_name = class_dir.name
            (split_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all image files recursively from the class directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_dir.rglob(ext)))
        
        print(f"Found {len(image_files)} images in {class_name}")
        
        if len(image_files) == 0:
            print(f"No images found in {class_name}, skipping...")
            continue
        
        # Shuffle the files randomly
        random.shuffle(image_files)
        
        # Calculate split indices
        total_files = len(image_files)
        test_size = int(0.15 * total_files)
        val_size = int(0.15 * total_files)
        train_size = total_files - test_size - val_size
        
        # Split the files
        train_files = image_files[:train_size]
        test_files = image_files[train_size:train_size + test_size]
        val_files = image_files[train_size + test_size:]
        
        print(f"{class_name} split: {len(train_files)} train, {len(test_files)} test, {len(val_files)} val")
        
        # Copy files to respective directories
        for files, split_name in [(train_files, 'train'), (test_files, 'test'), (val_files, 'val')]:
            dest_dir = split_dir / split_name / class_name
            for file_path in files:
                dest_path = dest_dir / file_path.name
                shutil.copy2(file_path, dest_path)
        
        print(f"Completed splitting {class_name}")
    
    print("\nData splitting completed!")
    print(f"Output directory: {split_dir.absolute()}")

if __name__ == "__main__":
    split_data()
