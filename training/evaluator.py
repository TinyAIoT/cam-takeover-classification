import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Evaluator:
    """Class for model evaluation, metrics calculation, and visualization"""
    
    def __init__(self, config, dataset_handler):
        """Initialize the Evaluator with configuration and dataset handler
        Args:
            config (dict): Dictionary containing configuration parameters
            dataset_handler (DatasetHandler): Instance of DatasetHandler for data access
        """
        self.model_name = config["model_name"]
        self.output_path = os.path.join(config["output_path"], self.model_name)
        self.dataset_handler = dataset_handler
        
    def evaluate_model(self, model, history, best_model=False, best_epoch=0, last_epoch=0, model_name=None):
        """Evaluate the model on test data
        Args:
            model (torch.nn.Module): The model to evaluate
            history (dict): Training history containing metrics
            best_model (bool): Whether to load the best model for evaluation
            best_epoch (int): Epoch number of the best model
            last_epoch (int): Last epoch number for evaluation
            model_name (str): Name of the model to use for saving results
        Returns:
            dict: Dictionary containing evaluation metrics such as test loss, accuracy, precision, recall, and F1 score
        """
        _, _, test_loader = self.dataset_handler.get_data_loaders()
        
        if model_name is None:
            model_name = self.model_name
        
        # Load best model if requested
        if best_model:
            print(f"Loading best model for tests from {os.path.join(self.output_path, model_name + '.pt')}")
            model.load_state_dict(torch.load(os.path.join(self.output_path, model_name + ".pt")))
            epoch = best_epoch
        else:
            epoch = last_epoch
            
        # Set model in evaluation mode
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)    
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # For confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        print(f"all labels: {all_labels[:10]}")
        class_names = self.dataset_handler.get_class_names()
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
        precision, recall = self._calculate_precision_recall(cm)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision}, Recall: {recall}")
        
        # Generate and save confusion matrix
        self._plot_confusion_matrix(cm, class_names)
        
        test_accuracy = correct / total
        print(f"Test Loss: {test_loss / len(self.dataset_handler.test_data):.4f}, Accuracy: {100 * test_accuracy:.2f}%")
        
        # Save metrics
        self._save_test_metrics(history, epoch, test_accuracy)
        
        return {
            'test_loss': test_loss / len(self.dataset_handler.test_data),
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def _calculate_precision_recall(self, cm):
        """Calculate precision and recall from confusion matrix
        Args:
            cm (numpy.ndarray): Confusion matrix as a 2D numpy array
        Returns:
            tuple: Precision and recall values
        """
        num_classes = np.shape(cm)[0]

        for j in range(num_classes):
            tp = np.sum(cm[j, j])
            fp = np.sum(cm[j, np.concatenate((np.arange(0, j), np.arange(j+1, num_classes)))])
            fn = np.sum(cm[np.concatenate((np.arange(0, j), np.arange(j+1, num_classes))), j])

            precision = tp/(tp+fp) if (tp+fp) > 0 else 0
            recall = tp/(tp+fn) if (tp+fn) > 0 else 0
        return precision, recall
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix
        Args:
            cm (numpy.ndarray): Confusion matrix as a 2D numpy array
            class_names (list): List of class names for labeling the matrix
        """
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, self.model_name + "_confusion_matrix.png"))
        plt.close()
        print(f"Confusion matrix saved to {os.path.join(self.output_path, self.model_name + '_confusion_matrix.png')}")
    
    def _save_test_metrics(self, history, epoch, test_accuracy):
        """Save test metrics to CSV
        Args:
            history (dict): Training history containing metrics
            epoch (int): Epoch number for which the metrics are saved
            test_accuracy (float): Test accuracy value
        """
        test_metrics = np.array([[history['accuracy'][epoch], history['val_accuracy'][epoch], test_accuracy]])
        np.savetxt(
            os.path.join(self.output_path, self.model_name + "_best-metrics.csv"), 
            test_metrics, 
            delimiter=',', 
            header='best_train_accuracy,best_val_accuracy,test_accuracy', 
            comments='', 
            fmt='%f'
        )
    
    def plot_training_history(self, history, initial_epochs=None):
        """Plot and save training history
        Args:
            history (dict): Training history containing loss and accuracy metrics
            initial_epochs (int, optional): Initial epochs for fine-tuning, if applicable
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        print(f"history: {history}")
        # Convert from tensor to numpy arrays for plotting
        train_loss = np.array(history['loss'])
        val_loss = np.array(history['val_loss'])
        train_accuracy = np.array(history['accuracy'])
        val_accuracy = np.array(history['val_accuracy'])

        # Save metrics to csv
        metrics = np.array([train_loss, val_loss, train_accuracy, val_accuracy])
        print("metrics shape: ")
        print(metrics.shape)
        np.savetxt(
            os.path.join(self.output_path, self.model_name + "_train-metrics.csv"), 
            metrics.T, 
            delimiter=',', 
            header='train_loss,val_loss,train_accuracy,val_accuracy', 
            comments='', 
            fmt='%f'
        )

        ax[0].plot(train_loss, label='Training Loss')
        ax[0].plot(val_loss, label='Validation Loss')
        if initial_epochs is not None:
            ymax = max(max(train_loss), max(val_loss)) * 1.1
            ax[0].set_ylim(0, ymax)
            ax[0].plot([initial_epochs,initial_epochs], [0,ymax], label='Start Fine Tuning')
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(train_accuracy, label='Training Accuracy')
        ax[1].plot(val_accuracy, label='Validation Accuracy')
        if initial_epochs is not None:
            ymin = min(min(train_accuracy), min(val_accuracy)) * 0.9
            ax[1].set_ylim(ymin, 1.0)  # Set y-limits for better visibility
            ax[1].plot([initial_epochs,initial_epochs], [ymin, 1.0], label='Start Fine Tuning')
        ax[1].set_title('Accuracy over epochs')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, self.model_name + "_history.png"))
        print(f"History plot saved to {os.path.join(self.output_path, self.model_name + '_history.png')}")
    
    def generate_evaluation_report(self, evaluation_results):
        """Generate a comprehensive evaluation report
        Args:
            evaluation_results (dict): Dictionary containing evaluation metrics
        Returns:
            str: Formatted evaluation report as a string
        """
        report = f"""
# Model Evaluation Report

## Model Information
- Model Name: {self.model_name}
- Model Type: {self.dataset_handler.get_class_names()}

## Test Results
- Test Loss: {evaluation_results['test_loss']:.4f}
- Test Accuracy: {evaluation_results['test_accuracy']:.4f} ({evaluation_results['test_accuracy']*100:.2f}%)
- Precision: {evaluation_results['precision']:.4f}
- Recall: {evaluation_results['recall']:.4f}
- F1 Score: {evaluation_results['f1_score']:.4f}

## Dataset Information
- Number of Classes: {len(self.dataset_handler.get_class_names())}
- Class Names: {self.dataset_handler.get_class_names()}
- Test Dataset Size: {len(self.dataset_handler.test_data)}
- Device Used: {self.dataset_handler.get_device()}
        """
        
        # Save report
        with open(os.path.join(self.output_path, self.model_name + "_evaluation_report.txt"), 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to {os.path.join(self.output_path, self.model_name + '_evaluation_report.txt')}")
        return report 