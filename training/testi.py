import pandas as pd

class FrameStackDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, img_height=24, img_width=24):
        """
        Dataset for loading frame stacks from CSV and images
        
        Args:
            csv_path: Path to CSV file with columns [frame1, frame2, ..., frame15, label]
            img_dir: Directory containing grayscale JPG images
            transform: Optional transform to apply to images
            img_height: Height to resize images to (default 24)
            img_width: Width to resize images to (default 24)
        """
        self.csv_data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
        # Assume CSV columns are: frame1, frame2, ..., frame15, label
        self.frame_columns = [f'frame{i+1}' for i in range(15)]
        
        # Basic transform for grayscale images
        self.base_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        
        # Load 15 frames
        frames = []
        for frame_col in self.frame_columns:
            img_path = os.path.join(self.img_dir, row[frame_col])
            # Load grayscale image
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            if self.transform:
                image = self.transform(image)
            else:
                image = self.base_transform(image)
            
            # Flatten the image (24x24 -> 576)
            frame_flat = image.view(-1)  # Flatten to 1D
            frames.append(frame_flat)
        
        # Stack frames: (15, 576)
        frame_stack = torch.stack(frames)
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return frame_stack, label

# Add this method to your ModelTrainer class
def load_frame_stack_data(self):
    """Load frame stack data from CSV files"""
    
    # Transform for grayscale images
    transform = transforms.Compose([
        transforms.Resize((24, 24)),
        transforms.ToTensor(),
        # Add any additional transforms like normalization if needed
    ])
    
    # Load datasets
    self.train_data = FrameStackDataset(
        csv_path=os.path.join(self.data_path, "train.csv"),
        img_dir=os.path.join(self.data_path, "images"),
        transform=transform
    )
    
    self.val_data = FrameStackDataset(
        csv_path=os.path.join(self.data_path, "val.csv"),
        img_dir=os.path.join(self.data_path, "images"),
        transform=transform
    )
    
    self.test_data = FrameStackDataset(
        csv_path=os.path.join(self.data_path, "test.csv"),
        img_dir=os.path.join(self.data_path, "images"),
        transform=transform
    )
    
    # For binary classification
    self.class_names = ['normal', 'anomaly']  # or whatever your classes are
    
    print("Class names: ", self.class_names)
    print("Train dataset size:", len(self.train_data))
    print("Validation dataset size:", len(self.val_data))
    print("Test dataset size:", len(self.test_data))

# Update the custom model to handle the correct input shape
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        # Input: (batch_size, 15, 576) where 576 = 24*24
        
        # Conv2D layers - need to reshape to (batch_size, 15, 24, 24) first
        self.conv1 = nn.Conv2d(15, 6, kernel_size=3, padding=0)  # (15, 24, 24) -> (6, 22, 22)
        self.conv2 = nn.Conv2d(6, 3, kernel_size=3, padding=0)   # (6, 22, 22) -> (3, 20, 20)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # (3, 20, 20) -> (3, 1, 1)
        
        # Dense layer for classification
        if num_classes == 2:
            self.classifier = nn.Linear(3, 1)  # Binary classification
        else:
            self.classifier = nn.Linear(3, num_classes)  # Multi-class
        
        self.num_classes = num_classes
    
    def forward(self, x):
        # Input shape: (batch_size, 15, 576)
        # Reshape to (batch_size, 15, 24, 24)
        batch_size = x.size(0)
        x = x.view(batch_size, 15, 24, 24)
        
        # Conv2D layers with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 3)
        
        # Final classification layer
        x = self.classifier(x)
        
        # Apply activation based on number of classes
        if self.num_classes == 2:
            x = torch.sigmoid(x)  # Binary classification
        
        return x