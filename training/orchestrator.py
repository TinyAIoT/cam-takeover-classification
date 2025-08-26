import os
import sys
from dataset_handler import DatasetHandler
from model_factory import ModelFactory
from learning_configurator import LearningConfigurator
from trainer import Trainer
from evaluator import Evaluator


class Orchestrator:
    """
    Advanced model trainer that properly handles transfer learning workflow:
    1. DatasetHandler - Dataset loading and preprocessing
    2. ModelFactory - Model creation and architecture selection
    3. LearningConfigurator - Handles transfer learning and fine tuning configuration (freeze/unfreeze backbone and head)
    4. Trainer - Generic training loop (used by both phases)
    5. Evaluator - Model evaluation and visualization
    """
    
    def __init__(self, config):
        """Initialize the Orchestrator with configuration parameters
        Args:
            config (dict): Dictionary containing configuration parameters for the transfer learning pipeline.
        """
        self.config = config
        self.model_name = config["model_name"]
        self.output_path = os.path.join(config["output_path"], self.model_name)
        self.profile_mode = config.get("profile", False)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        # Initialize all components
        self.dataset_handler = DatasetHandler(config)
        self.model_factory = ModelFactory(config)
        self.learning_configurator = LearningConfigurator()
        self.trainer = Trainer(config, self.dataset_handler)
        self.evaluator = Evaluator(config, self.dataset_handler)
        
        # Model and training state
        self.model = None
        self.transfer_learning_history = None
        self.fine_tuning_history = None
        self.best_epoch = 0
        self.last_epoch = 0
        self.initial_epochs = 0
        
    def run_transfer_learning_pipeline(self):
        """Run the complete transfer learning pipeline (transfer learning + optional fine-tuning)
        Returns:
            dict: Dictionary containing the trained model, training history, evaluation results, best epoch, and last epoch.
        """
        print("=" * 70)
        print(f"Starting transfer learning pipeline for {self.model_name}")
        print("=" * 70)
        
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        self.dataset_handler.load_data()
        self.dataset_handler.preprocess_data()
        
        # Step 2: Create model
        print("\n2. Creating model...")
        num_classes = len(self.dataset_handler.get_class_names())
        device = self.dataset_handler.get_device()
        self.model = self.model_factory.create_model(num_classes, device)
        
        # Print model information
        print(f"\nModel: {self.model_factory.get_model_info()}")
        
        # Step 3: Transfer Learning Phase
        print("\n3. Starting Transfer Learning Phase...")
        self.model = self.learning_configurator.prepare_model_for_transfer_learning(self.model)
        
        # Train classification head
        self.transfer_learning_history, tl_best_epoch, tl_last_epoch = self.trainer.train_model(
            self.model, 
            epochs=self.config["transfer_learning_epochs"], 
            learning_rate=self.config["transfer_learning_rate"], 
            weight_decay=self.config["transfer_weight_decay"],
            phase_name='transfer_learning',
            checkpoint_prefix=f"{self.model_name}_transfer_learning",
            profile=self.profile_mode
        )

        print(self.trainer.get_training_summary("Transfer-learning"))
        
        print(self.config)
        
        # Step 4: Fine-tuning Phase (if enabled)
        if self.config.get("fine_tune", False):
            print("\n4. Starting Fine-tuning Phase...")
            self.model = self.learning_configurator.prepare_model_for_fine_tuning(self.model, self.config["num_unfreezed_feature_layers"])
            
            # Reset trainer history for fine-tuning
            self.trainer.reset_history()
            
            # Train with fine-tuning
            self.fine_tuning_history, ft_best_epoch, ft_last_epoch = self.trainer.train_model(
                self.model,
                epochs=self.config['fine_tune_epochs'],
                learning_rate=self.config['fine_tuning_learning_rate'],
                weight_decay=self.config['fine_tuning_weight_decay'],
                phase_name='fine_tuning',
                checkpoint_prefix=f"{self.model_name}_fine_tuning",
                profile=self.profile_mode
            )
            
            print(self.trainer.get_training_summary("Fine-tuning"))
            
            # Update best epoch for evaluation
            self.best_epoch = ft_best_epoch
            self.last_epoch = ft_last_epoch
        else:
            print("\n4. Fine-tuning disabled - skipping fine-tuning phase")
            self.fine_tuning_history = None
            self.best_epoch = tl_best_epoch
            self.last_epoch = tl_last_epoch
        
        # Step 5: Evaluate final model
        print("\n5. Evaluating final model...")
        # concat histories if fine-tuning was done
        final_history = {key: self.transfer_learning_history[key] + self.fine_tuning_history[key] for key in self.transfer_learning_history} if self.fine_tuning_history else self.transfer_learning_history
        self.best_epoch = tl_last_epoch + self.best_epoch if self.fine_tuning_history else tl_best_epoch
        self.last_epoch = tl_last_epoch + ft_last_epoch if self.fine_tuning_history else tl_last_epoch
        self.initial_epochs = tl_last_epoch
        print(f"Initial epochs: {self.initial_epochs}")


        evaluation_results = self.evaluator.evaluate_model(
            self.model, 
            final_history, 
            best_model=True, 
            best_epoch=self.best_epoch, 
            last_epoch=self.last_epoch,
            model_name=f"{self.model_name}_fine_tuning" if self.fine_tuning_history else f"{self.model_name}_transfer_learning"
        )
        
        # Step 6: Generate visualizations and reports
        print("\n6. Generating visualizations and reports...")
        self.evaluator.plot_training_history(final_history, self.initial_epochs if self.fine_tuning_history else None)
        self.evaluator.generate_evaluation_report(evaluation_results)
        
        # Generate combined training report
        self._generate_combined_training_report(evaluation_results)
        
        # Print final summary
        print("\n" + "=" * 70)
        print("TRANSFER LEARNING PIPELINE COMPLETED")
        print("=" * 70)

        if self.fine_tuning_history:
            print(self.trainer.get_training_summary("Fine-tuning"))
        print(f"\nFinal Test Accuracy: {evaluation_results['test_accuracy']:.4f} ({evaluation_results['test_accuracy']*100:.2f}%)")
        print(f"Final F1 Score: {evaluation_results['f1_score']:.4f}")
        print(f"Output saved to: {self.output_path}")
        
        return {
            'model': self.model,
            'transfer_learning_history': self.transfer_learning_history,
            'fine_tuning_history': self.fine_tuning_history,
            'evaluation_results': evaluation_results,
            'best_epoch': self.best_epoch,
            'last_epoch': self.last_epoch
        }
    
    def run_transfer_learning_only(self, model=None):
        """Run only the transfer learning phase
        Args:
            model (optional): Pre-trained model to use for transfer learning. If None, a new model will be created.
        Returns:
            dict: Dictionary containing the trained model, transfer learning history, best epoch, and last epoch.
        """
        if model is not None:
            self.model = model

        if self.model is None:
            # Step 1: Load and preprocess data
            print("\n1. Loading and preprocessing data...")
            self.dataset_handler.load_data()
            self.dataset_handler.preprocess_data()
            
            # Step 2: Create model
            print("\n2. Creating model...")
            num_classes = len(self.dataset_handler.get_class_names())
            device = self.dataset_handler.get_device()
            self.model = self.model_factory.create_model(num_classes, device)
            
            # Print model information
            print(f"\nModel: {self.model_factory.get_model_info()}")
            # raise ValueError("Model not created. Run run_transfer_learning_pipeline() first or create model manually.")
        
        print("Running transfer learning phase only...")
        self.model = self.learning_configurator.prepare_model_for_transfer_learning(self.model)
        self.transfer_learning_history, self.best_epoch, self.last_epoch = self.trainer.train_model(
            self.model, 
            epochs=self.config["transfer_learning_epochs"],
            learning_rate=self.config["transfer_learning_rate"], 
            weight_decay=self.config["transfer_weight_decay"],
            phase_name='transfer_learning',
            checkpoint_prefix=f"{self.model_name}_transfer_learning",
            profile=self.profile_mode
        )
        print(self.trainer.get_training_summary("Transfer-learning"))
        
        return {
            'model': self.model,
            'transfer_learning_history': self.transfer_learning_history,
            'best_epoch': self.best_epoch,
            'last_epoch': self.last_epoch
        }
    
    
    def run_fine_tuning_only(self, load_transfer_learning_checkpoint=True, model=None):
        """Run only the fine-tuning phase
        Args:
            load_transfer_learning_checkpoint (bool): Whether to load the best transfer learning checkpoint before fine-tuning.
            model (optional): Pre-trained model to use for fine-tuning. If None, a new model will be created.
        Returns:
            dict: Dictionary containing the trained model, fine-tuning history, best epoch, and last epoch.
        """
        if model is not None:
            self.model = model
        
        if self.model is None:
            # Step 1: Load and preprocess data
            print("\n1. Loading and preprocessing data...")
            self.dataset_handler.load_data()
            self.dataset_handler.preprocess_data()
            
            # Step 2: Create model
            print("\n2. Creating model...")
            num_classes = len(self.dataset_handler.get_class_names())
            device = self.dataset_handler.get_device()
            self.model = self.model_factory.create_model(num_classes, device)
            
            # Print model information
            print(f"\nModel: {self.model_factory.get_model_info()}")
        
        if load_transfer_learning_checkpoint:
            # Load the best transfer learning checkpoint
            self.trainer.load_checkpoint(self.model, 
                                         os.path.join(self.output_path, f"{self.model_name}_transfer_learning.pt"))
        
        print("Running fine-tuning phase only...")
        self.model = self.learning_configurator.prepare_model_for_fine_tuning(self.model, self.config["num_unfreezed_feature_layers"])
                
        # Reset trainer history for fine-tuning
        self.trainer.reset_history()
        
        # Train with fine-tuning
        self.fine_tuning_history, self.best_epoch, self.last_epoch = self.trainer.train_model(
                self.model,
                epochs=self.config['fine_tune_epochs'],
                learning_rate=self.config['fine_tuning_learning_rate'],
                weight_decay=self.config['fine_tuning_weight_decay'],
                phase_name='fine_tuning',
                checkpoint_prefix=f"{self.model_name}_fine_tuning",
                profile=self.profile_mode
            )
        
        print(self.trainer.get_training_summary("Fine-tuning"))

        
        return {
            'model': self.model,
            'fine_tuning_history': self.fine_tuning_history,
            'best_epoch': self.best_epoch,
            'last_epoch': self.last_epoch
        }
    
    def evaluate_only(self, model_path=None, model=None):
        """Run only the evaluation phase
        Args:
            model_path (optional): Path to the model checkpoint to load for evaluation. If None, uses the current model.
            model (optional): Pre-trained model to use for evaluation. If None, uses the current model.
        Returns:
            dict: Dictionary containing evaluation results including test loss, accuracy, precision, recall, and F1 score.
        """
        if model is not None:
            self.model = model
        
        if self.model is None:
            raise ValueError("Model not created. Run run_transfer_learning_pipeline() first or create model manually.")
        
        if model_path:
            self.trainer.load_checkpoint(self.model, model_path)
        
        print("Running evaluation phase only...")
        final_history = {key: self.transfer_learning_history[key] + self.fine_tuning_history[key] for key in self.transfer_learning_history} if self.fine_tuning_history else self.transfer_learning_history
        evaluation_results = self.evaluator.evaluate_model(
            self.model, 
            final_history or {'accuracy': [], 'val_accuracy': []}, 
            best_model=True, 
            best_epoch=self.best_epoch + self.initial_epochs if self.fine_tuning_history else self.best_epoch, 
            last_epoch=self.last_epoch + self.initial_epochs if self.fine_tuning_history else self.last_epoch,
        )
        
        self.evaluator.plot_training_history(final_history or {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}, self.initial_epochs if self.fine_tuning_history else None)
        self.evaluator.generate_evaluation_report(evaluation_results)
        
        return evaluation_results
    
    def _generate_combined_training_report(self, evaluation_results):
        """Generate a comprehensive report combining both phases
        Args:
            evaluation_results (dict): Dictionary containing evaluation results including test loss, accuracy, precision, recall, and F1 score.
        Returns:
            str: Formatted report string containing model information, training history, and evaluation results.
        """
        
        report = f"""
# Complete Transfer Learning Pipeline Report

## Model Information
- Model Name: {self.model_name}
- Model Type: {self.model_factory.get_model_info()}
- Classes: {self.dataset_handler.get_class_names()}

## Final Results
- Test Loss: {evaluation_results['test_loss']:.4f}
- Test Accuracy: {evaluation_results['test_accuracy']:.4f} ({evaluation_results['test_accuracy']*100:.2f}%)
- Precision: {evaluation_results['precision']:.4f}
- Recall: {evaluation_results['recall']:.4f}
- F1 Score: {evaluation_results['f1_score']:.4f}

## Dataset Information
- Number of Classes: {len(self.dataset_handler.get_class_names())}
- Class Names: {self.dataset_handler.get_class_names()}
- Train/Val/Test sizes: {len(self.dataset_handler.train_data)}/{len(self.dataset_handler.val_data)}/{len(self.dataset_handler.test_data)}
- Device Used: {self.dataset_handler.get_device()}
        """
                
        # Save report
        with open(os.path.join(self.output_path, self.model_name + "_complete_pipeline_report.txt"), 'w') as f:
            f.write(report)
        
        print(f"Complete pipeline report saved to {os.path.join(self.output_path, self.model_name + '_complete_pipeline_report.txt')}")
        return report
    
    def get_component_info(self):
        """Get information about all components
        Returns:
            dict: Dictionary containing information about dataset handler, model factory, transfer learning config, and fine-tuning config.
        """
        info = {
            'dataset_handler': {
                'class_names': self.dataset_handler.get_class_names(),
                'device': self.dataset_handler.get_device(),
                'train_size': len(self.dataset_handler.train_data) if self.dataset_handler.train_data else 0,
                'val_size': len(self.dataset_handler.val_data) if self.dataset_handler.val_data else 0,
                'test_size': len(self.dataset_handler.test_data) if self.dataset_handler.test_data else 0
            },
            'model_factory': {
                'model_type': self.model_factory.model_type,
                'model_info': self.model_factory.get_model_info()
            },
            'transfer_learning': {
                'training_epochs': self.config['transfer_learning_epochs'],
                'learning_rate': self.config['transfer_learning_rate'],
                'patience': self.config["patience"],
                'weight_decay': self.config["transfer_weight_decay"]
            },
            'fine_tuning': {
                'fine_tune_enabled': self.config["fine_tune"],
                'fine_tune_epochs': self.config["fine_tune_epochs"],
                'fine_tuning_learning_rate': self.config["fine_tuning_learning_rate"],
                'fine_tuning_weight_decay': self.config["fine_tuning_weight_decay"],
                'unfrozen_layers': self.config["num_unfreezed_feature_layers"]
            }
        }
        return info
    
    def print_component_info(self):
        """Print information about all components"""
        info = self.get_component_info()
        
        print("\n" + "=" * 70)
        print("COMPONENT INFORMATION")
        print("=" * 70)
        
        print(f"\nDataset Handler:")
        print(f"  Classes: {info['dataset_handler']['class_names']}")
        print(f"  Device: {info['dataset_handler']['device']}")
        print(f"  Train/Val/Test sizes: {info['dataset_handler']['train_size']}/{info['dataset_handler']['val_size']}/{info['dataset_handler']['test_size']}")
        
        print(f"\nModel Factory:")
        print(f"  Model Type: {info['model_factory']['model_type']}")
        print(f"  Model Info: {info['model_factory']['model_info']}")
        
        print(f"\nTransfer Learner:")
        print(f"  Training epochs: {info['transfer_learning']['training_epochs']}")
        print(f"  Learning rate: {info['transfer_learning']['learning_rate']}")
        print(f"  Patience: {info['transfer_learning']['patience']}")
        print(f"  Weight decay: {info['transfer_learning']['weight_decay']}")
        
        print(f"\nFine Tuner:")
        print(f"  Fine-tuning enabled: {info['fine_tuning']['fine_tune_enabled']}")
        print(f"  Fine-tuning epochs: {info['fine_tuning']['fine_tune_epochs']}")
        print(f"  Fine-tuning learning rate: {info['fine_tuning']['fine_tuning_learning_rate']}")
        print(f"  Unfrozen layers: {info['fine_tuning']['unfrozen_layers']}")
        
        print("=" * 70) 