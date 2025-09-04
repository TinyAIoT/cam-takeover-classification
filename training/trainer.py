import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import json


class Trainer:
    """Generic training class that can be used for both transfer learning and fine-tuning"""
    
    def __init__(self, config, dataset_handler):
        """ Initialize the Trainer with configuration and dataset handler
        Args:
            config (dict): Configuration dictionary containing training parameters
            dataset_handler (DatasetHandler): Instance of DatasetHandler to access data loaders
        """
        self.model_name = config["model_name"]
        self.output_path = os.path.join(config["output_path"], self.model_name)
        
        self.dataset_handler = dataset_handler
        
        # Training state
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        self.best_epoch = 0
        self.last_epoch = 0
        
    def train_model(self, model, epochs=0, learning_rate=0, weight_decay=0, patience=0, phase_name='training', checkpoint_prefix='', profile=False):
        """
        Generic training method that can be used for both transfer learning and fine-tuning
        
        Args:
            model: The model to train
            epochs (int): Number of epochs to train
            learning_rate (float): Learning rate for the optimizer
            weight_decay (float): Weight decay for the optimizer
            phase_name (str): Name of the training phase (e.g., 'training', 'fine-tuning')
            checkpoint_prefix (str): Prefix for the checkpoint files
            profile (bool): Whether to enable profiling mode (default: False)
        Returns:
            history (dict): Training history containing loss and accuracy metrics
            best_epoch (int): The epoch number of the best model
            last_epoch (int): The last epoch number completed                   
        """
        # pretty print variables types and values
        print(f"Training model: {self.model_name}\n"
                f"Phase: {phase_name}\n"
                f"Epochs: {epochs}\n"
                f"Learning Rate: {learning_rate}\n"
                f"Weight Decay: {weight_decay}\n"
                f"Checkpoint Prefix: {checkpoint_prefix}\n"
                f"Output Path: {self.output_path}\n"
                f"Device: {self.dataset_handler.get_device()}\n"
                f"Train Data Size: {len(self.dataset_handler.train_data)}\n"
                f"Validation Data Size: {len(self.dataset_handler.val_data)}\n"
                )
        
        train_loader, val_loader, _ = self.dataset_handler.get_data_loaders()
        
        print("Train loader structure:")
        self.dataset_handler.print_data_loader_structure(train_loader)
        print("Validation loader structure:")
        self.dataset_handler.print_data_loader_structure(val_loader)

        # Setup training components
        criterion = nn.CrossEntropyLoss()
        
        # Only optimize trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting {phase_name} for {epochs} epochs...")

        for epoch in range(epochs):
            print(f'\n{phase_name.title()} Epoch {epoch + 1}/{epochs}')
            
            # Training phase
            if not profile:
                # Normal training mode
                train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer, phase_name)
            else:
                # Profiling mode - runs limited steps and records a trace
                print("Profiling mode enabled: training will stop after a limited number of steps")
                print("This is for profiling purposes only, not for full training")
                print("To run full training, set profile=False")
                train_loss, train_acc = self.profile_train_epoch(model, train_loader, criterion, optimizer, phase_name, scheduler)
                
            scheduler.step()
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # Record history
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc.cpu())
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc.cpu())
            
            # Print progress
            print('-' * 50)
            print(f'{phase_name.title()} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print(f'{phase_name.title()} - Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
                        
            # Checkpointing and early stopping
            if val_loss < best_val_loss:
                self.best_epoch = epoch
                print(f"New best {phase_name} model! Saving checkpoint at epoch {self.best_epoch+1}")
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(model, epoch, checkpoint_prefix, phase_name)
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    self.last_epoch = epoch
                    print(f'{phase_name.title()} early stopping triggered after {patience} epochs without improvement')
                    break
            self.last_epoch = epoch
            print(f'Current best {phase_name} model at epoch {self.best_epoch + 1} with validation loss {best_val_loss:.4f}')
            print('-' * 50)

        print("\n" + "=" * 50)
        print(f"{phase_name.title()} completed!")
        print(f"Best {phase_name} model saved at epoch {self.best_epoch + 1}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 50)
        return self.history, self.best_epoch, self.last_epoch
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, phase_name, scheduler=None, callbacks=None):
        """Train for one epoch
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer for updating model parameters
            phase_name (str): Name of the training phase (e.g., 'training', 'fine-tuning')
            scheduler: Learning rate scheduler (optional)
            callbacks: List of callback functions to execute during training (optional)
        Returns:
            epoch_loss (float): Average loss for the epoch
            epoch_acc (float): Average accuracy for the epoch
        """
        model.train()
        running_loss = 0.0
        running_corrects = 0
        device = self.dataset_handler.get_device()
        
        for images, labels in tqdm(train_loader, desc=phase_name.title(), unit='batch'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            if callbacks is not None:
                for callback in callbacks:
                    callback()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.dataset_handler.train_data)
        epoch_acc = running_corrects.float() / len(self.dataset_handler.train_data)
        
        return epoch_loss, epoch_acc
    
    def profile_train_epoch(self, model, train_loader, criterion, optimizer, phase_name, scheduler=None, callbacks=None, wait=1, warmup=4, active=8, repeat=1):
        """Profiling variant of _train_epoch: runs limited steps and records a trace.
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer for updating model parameters
            phase_name (str): Name of the training phase (e.g., 'training', 'fine-tuning')
            scheduler: Learning rate scheduler (optional)
            callbacks: List of callback functions to execute during training (optional)
            wait, warmup, active, repeat: Profiling parameters
        Returns:
            epoch_loss (float): Average loss for the epoch
            epoch_acc (float): Average accuracy for the epoch
        """
        model.train()
        running_loss = 0.0
        running_corrects = 0
        device = self.dataset_handler.get_device()

        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=4, active=8, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(self.output_path, "profiler")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i, (images, labels) in enumerate(tqdm(train_loader, desc=phase_name.title(), unit="batch")):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                if callbacks is not None:
                    for callback in callbacks:
                        callback()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

                prof.step()  # advance profiler schedule

                if (i + 1) >= wait + warmup + active * repeat:
                    break

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        epoch_size = len(self.dataset_handler.train_data)
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_corrects.float() / epoch_size
        return epoch_loss, epoch_acc        
    
    def _validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch
        Args:
            model: The model to validate
            val_loader: DataLoader for validation data
            criterion: Loss function
        Returns:
            epoch_loss (float): Average loss for the epoch
            epoch_acc (float): Average accuracy for the epoch
        """
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        device = self.dataset_handler.get_device()
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_loss += loss.item() * images.size(0)

        epoch_loss = val_loss / len(self.dataset_handler.val_data)
        epoch_acc = val_corrects.float() / len(self.dataset_handler.val_data)
        
        return epoch_loss, epoch_acc
    
    
    def _save_checkpoint(self, model, epoch, checkpoint_prefix, phase_name):
        """Save model checkpoint
        Args:
            model: The model to save
            epoch (int): Current epoch number
            checkpoint_prefix (str): Prefix for the checkpoint files
            phase_name (str): Name of the training phase (e.g., 'training', 'fine-tuning')
        """
        # Save full model
        torch.save(model, os.path.join(self.output_path, f"{checkpoint_prefix}.pth"))
        
        # Save state dict
        torch.save(model.state_dict(), os.path.join(self.output_path, f"{checkpoint_prefix}.pt"))
        
        # Save training info
        checkpoint_info = {
            'epoch': epoch,
            'model_name': self.model_name,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'phase': phase_name
        }
        
        torch.save(checkpoint_info, os.path.join(self.output_path, f"{checkpoint_prefix}_info.pt"))
    
    
    def get_training_summary(self, phase_name="Training"):
        """Get a summary of the training results
        Args:
            phase_name (str): Name of the training phase (e.g., 'Training', 'Fine-tuning')
        Returns:
            str: Summary string containing training statistics
        """
        if not self.history['loss']:
            return f"No {phase_name.lower()} history available"
        
        best_val_acc = max(self.history['val_accuracy'])
        final_train_acc = self.history['accuracy'][-1]
        final_val_acc = self.history['val_accuracy'][-1]
        
        summary = f"""
{phase_name.title()} Summary:
- Total epochs: {len(self.history['loss'])}
- Best epoch: {self.best_epoch + 1}
- Best validation accuracy: {best_val_acc:.4f}
- Final training accuracy: {final_train_acc:.4f}
- Final validation accuracy: {final_val_acc:.4f}
        """
        return summary
    
    def get_history(self):
        """Get training history
        Returns:
            dict: Training history containing loss and accuracy metrics
        """
        return self.history
    
    def get_best_epoch(self):
        """Get the best epoch number
        Returns:
            int: The epoch number of the best model
        """
        return self.best_epoch
    
    def reset_history(self):
        """Reset training history (useful when switching between phases)"""
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        self.best_epoch = 0
        self.last_epoch = 0 