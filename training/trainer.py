import sys
print(sys.version)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.ops import box_iou
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# Force PyTorch to use CPU only
# torch.cuda.is_available = lambda: False
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Get the versions of PyTorch and torchvision
torch_version = torch.__version__
torchvision_version = torchvision.__version__

# Log the versions
print(f"Active PyTorch version: {torch_version}")
print(f"Active torchvision version: {torchvision_version}")
		

# for moving data into GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    # if torch.cuda.is_available:
    #     return torch.device("cuda")
    # else:
    return torch.device("cpu")

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data.to(device, non_blocking=True)

# def collate_fn(batch):
#     images, targets = zip(*batch)
#     images = list(images)
#     targets = [{k: v for k, v in t.items()} for t in targets]
#     return images, targets

# for loading in the device (GPU if available else CPU)
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
        """Number of batches"""
        return len(self.dl)
    
    def getDataLoader(self):
        return self.dl

class ModelTrainer:
    def __init__(self, config):
        self.img_height = config["img_height"]
        self.img_width = config["img_width"]
        self.batch_size = config["batch_size"]
        self.training_epochs = config["training_epochs"]
        self.patience = config["patience"]
        self.data_path = config["dataset_path"]
        self.model_name = config["model_name"]
        self.model_type = config["model_type"]
        self.num_workers = config["num_workers"]
        self.output_path = os.path.join(config["output_path"], self.model_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_names = None
        self.model = None
        self.model_task = config["model_task"]
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        self.device = get_default_device()
        self.use_engine = True
        print("Using device:", self.device)
        
    def load_data(self):
        # Custom transform: crop (img_height, img_width) from the bottom center of the image
        class BottomCenterCrop:
            def __init__(self, height, width):
                self.height = height
                self.width = width

            def __call__(self, img):
                w, h = img.size
                left = max((w - self.width) // 2, 0)
                top = max(h - self.height, 0)
                right = left + self.width
                bottom = top + self.height
                return img.crop((left, top, right, bottom))

        transform = transforms.Compose([
            BottomCenterCrop(self.img_height, self.img_width),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_data = datasets.ImageFolder(os.path.join(self.data_path, "train"), transform)
        self.val_data = datasets.ImageFolder(os.path.join(self.data_path, "val"), transform)
        self.test_data = datasets.ImageFolder(os.path.join(self.data_path, "test"), transform)
        self.class_names = self.train_data.classes

        print("Class names: ")
        print(self.class_names)
        print("Train dataset size:", len(self.train_data))
        print("Validation dataset size:", len(self.val_data))
        print("Test dataset size:", len(self.test_data))
    
    def preprocess_data(self):
        # Utilize PyTorch's DataLoader for batching
        self.train_loader = DeviceDataLoader(DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers), self.device)
        self.val_loader = DeviceDataLoader(DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers), self.device)
        self.test_loader = DeviceDataLoader(DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers), self.device)

    def convert_relu6_to_relu(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU6):
                inplace = child.inplace  # Get the inplace parameter, which is default=False for ReLU6
                setattr(model, child_name, nn.ReLU(inplace=inplace))
            else:
                self.convert_relu6_to_relu(child)
        # save the model to a new file
        return model
    
    def create_model(self):
        num_classes = len(self.class_names)

        # squeezenet:
        if self.model_type == "squeezenet":
            print("create squeezenet")
            self.model = torchvision.models.squeezenet1_1(weights='DEFAULT')
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Modify the classifier for smaller input size (96x96)
            # The original classifier expects 13x13 feature maps, but with 96x96 input we get 6x6
            # We need to adjust the adaptive pooling and final conv layer
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.model.features[0] = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

            # Normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        # custom:
        elif self.model_type == "custom":
            print("create custom model")
            
            class CustomModel(nn.Module):
                def __init__(self, num_classes):
                    super(CustomModel, self).__init__()
                    # Reshape will be handled in forward pass
                    # Input: (batch_size, 20, 64) -> reshape to (batch_size, 20, 8, 8)
                    
                    # Conv2D layers
                    self.conv1 = nn.Conv2d(20, 6, kernel_size=3, padding=0)  # (20, 8, 8) -> (6, 6, 6)
                    self.conv2 = nn.Conv2d(6, 3, kernel_size=3, padding=0)   # (6, 6, 6) -> (3, 4, 4)
                    
                    # Global Average Pooling
                    self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # (3, 4, 4) -> (3, 1, 1)
                    
                    # Dense layer for binary classification
                    if num_classes == 2:
                        self.classifier = nn.Linear(3, 1)  # Binary classification with sigmoid
                    else:
                        self.classifier = nn.Linear(3, num_classes)  # Multi-class with softmax
                    
                    self.num_classes = num_classes
                
                def forward(self, x):
                    # Input shape: (batch_size, 15, 24x24)
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
                    # For multi-class, softmax is typically applied in loss function
                    
                    return x
            
            self.model = CustomModel(num_classes)
        else:
            raise ValueError("Unsupported model type. Please use 'shufflenet' or 'mobilenet'.")
        # self.model = to_device(self.model, self.device) 
        self.model = self.model.to(self.device)
        
        # Check if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs for training.")
            self.model = nn.DataParallel(self.model)  # Wrap the model with DataParallel        
                
    def train_model(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Alternativ optimizer and scheduler
        # optimizer = optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        best_val_loss = float('inf')
        best_acc = 0.0
        patience_counter = 0

        for epoch in range(self.training_epochs):
            print(f'Epoch {epoch + 1}/{self.training_epochs}')
            self.model.train()

            running_loss = 0.0
            running_corrects = 0
            for images, labels in tqdm(self.train_loader, desc='Training', unit='batch'):
                # images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            scheduler.step()
            epoch_loss = running_loss / len(self.train_data)
            epoch_acc = running_corrects.double() / len(self.train_data)
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_acc.cpu())
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Validation phase
            val_loss, val_acc = self.validate_model()
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc.cpu())
            print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model, os.path.join(self.output_path, self.model_name + '.pth'))
                torch.save(self.model.state_dict(), os.path.join(self.output_path, self.model_name + ".pt"))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print('Early stopping!')
                    break

    def validate_model(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                # images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_loss += loss.item() * images.size(0)
                

        return val_loss / len(self.val_data), val_corrects.double() / len(self.val_data) 
    
    def validate_detection_model(self):
        val_running_loss = 0.0
        val_pred_boxes = []
        val_true_boxes = []
        val_scores = []
        val_labels = []
        val_ious = []
        val_tp = 0
        val_fp = 0
        val_fn = 0
        val_coco_gt_annotations = []
        val_coco_dt_annotations = []
        val_annotation_id = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                self.model.train()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_running_loss += losses.item()

                self.model.eval()
                outputs = self.model(images)
                for i, output in enumerate(outputs):
                    pred_boxes = output['boxes']
                    true_boxes = targets[i]['boxes']
                    scores = output['scores']
                    labels = output['labels']
                    val_pred_boxes.append(pred_boxes)
                    val_true_boxes.append(true_boxes)
                    val_scores.append(scores)
                    val_labels.append(labels)

                    # Calculate IoU
                    iou = box_iou(pred_boxes, true_boxes)
                    val_ious.append(iou.mean().item())

                    # Prepare COCO format for ground truth
                    for box in true_boxes:
                        val_coco_gt_annotations.append({
                            'id': val_annotation_id,
                            'image_id': i,
                            'category_id': 1,  # Assuming all objects belong to the same class
                            'bbox': [box[0].item(), box[1].item(), box[2].item() - box[0].item(), box[3].item() - box[1].item()],
                            'area': (box[2].item() - box[0].item()) * (box[3].item() - box[1].item()),
                            'iscrowd': 0
                        })
                        val_annotation_id += 1

                    # Prepare COCO format for detections
                    for box, score in zip(pred_boxes, scores):
                        val_coco_dt_annotations.append({
                            'image_id': i,
                            'category_id': 1,  # Assuming all objects belong to the same class
                            'bbox': [box[0].item(), box[1].item(), box[2].item() - box[0].item(), box[3].item() - box[1].item()],
                            'score': score.item()
                        })

                    # Calculate precision and recall
                    for j in range(len(pred_boxes)):
                        if iou[j].max() > 0.5:  # IoU threshold for true positive
                            val_tp += 1
                        else:
                            val_fp += 1
                    val_fn += len(true_boxes) - val_tp

        val_epoch_loss = val_running_loss / len(self.val_loader)
        val_mean_iou = sum(val_ious) / len(val_ious) if val_ious else 0

        # Create COCO objects
        val_coco_gt = COCO()
        val_coco_gt.dataset = {
            'images': [{'id': i} for i in range(len(val_pred_boxes))],
            'annotations': val_coco_gt_annotations,
            'categories': [{'id': 1, 'name': 'object'}]
        }
        val_coco_gt.createIndex()

        val_coco_dt = val_coco_gt.loadRes(val_coco_dt_annotations)

        # Create COCOeval object
        val_coco_eval = COCOeval(val_coco_gt, val_coco_dt, 'bbox')
        val_coco_eval.params.imgIds = list(range(len(val_pred_boxes)))
        val_coco_eval.evaluate()
        val_coco_eval.accumulate()
        val_coco_eval.summarize()

        val_mean_ap = val_coco_eval.stats[0]  # mAP at IoU=0.50:0.95

        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0

        return val_epoch_loss, val_mean_ap, val_mean_iou, val_precision, val_recall

    def evaluate_model(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in self.test_loader:
                # images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Loss: {test_loss / len(self.test_data):.4f}, Accuracy: {100 * correct / total:.2f}%")

    def plot_history(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Convert from tensor to numpy arrays for plotting
        train_loss = np.array(self.history['loss'])
        val_loss = np.array(self.history['val_loss'])
        train_accuracy = np.array(self.history['accuracy'])
        val_accuracy = np.array(self.history['val_accuracy'])

        ax[0].plot(train_loss, label='Training Loss')
        ax[0].plot(val_loss, label='Validation Loss')
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(train_accuracy, label='Training Accuracy')
        ax[1].plot(val_accuracy, label='Validation Accuracy')
        ax[1].set_title('Accuracy over epochs')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, self.model_name + "_history.png"))
        print(f"History plot saved to {os.path.join(self.output_path, self.model_name + '_history.png')}")
    
    def plot_detection_history(self):
        fig, ax = plt.subplots(1, 5, figsize=(25, 5))

        # Convert from tensor to numpy arrays for plotting
        train_loss = np.array(self.history['loss'])
        val_loss = np.array(self.history['val_loss'])
        train_map = np.array(self.history['mAP'])
        val_map = np.array(self.history['val_mAP'])
        train_iou = np.array(self.history['IoU'])
        val_iou = np.array(self.history['val_IoU'])
        train_precision = np.array(self.history['precision'])
        val_precision = np.array(self.history['val_precision'])
        train_recall = np.array(self.history['recall'])
        val_recall = np.array(self.history['val_recall'])

        ax[0].plot(train_loss, label='Training Loss')
        ax[0].plot(val_loss, label='Validation Loss')
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(train_map, label='Training mAP')
        ax[1].plot(val_map, label='Validation mAP')
        ax[1].set_title('mAP over epochs')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('mAP')
        ax[1].legend()

        ax[2].plot(train_iou, label='Training IoU')
        ax[2].plot(val_iou, label='Validation IoU')
        ax[2].set_title('IoU over epochs')
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('IoU')
        ax[1].legend()

        ax[3].plot(train_precision, label='Training Precision')
        ax[3].plot(val_precision, label='Validation Precision')
        ax[3].set_title('Precision over epochs')
        ax[3].set_xlabel('Epoch')
        ax[3].set_ylabel('Precision')
        ax[1].legend()

        ax[4].plot(train_recall, label='Training Recall')
        ax[4].plot(val_recall, label='Validation Recall')
        ax[4].set_title('Recall over epochs')
        ax[4].set_xlabel('Epoch')
        ax[4].set_ylabel('Recall')
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, self.model_name + "_history.png"))
        plt.close()
        print(f"History plot saved to {os.path.join(self.output_path, self.model_name + '_history.png')}")