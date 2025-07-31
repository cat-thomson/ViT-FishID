import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

from vit_model import ViTForFishClassification
from ema_teacher import EMATeacher
from data_loader import create_fish_dataloaders
from utils import accuracy, load_checkpoint, get_device


class ModelEvaluator:
    """
    Comprehensive model evaluation for ViT-Fish classification.
    """
    
    def __init__(
        self,
        model: ViTForFishClassification,
        class_names: List[str],
        device: str = 'cuda'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            class_names: List of class names
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.class_names = class_names
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(
        self, 
        data_loader: DataLoader,
        save_dir: str = './evaluation_results'
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation on dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            save_dir: Directory to save results
            
        Returns:
            Dictionary of evaluation metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        total_correct = 0
        total_samples = 0
        
        print("Evaluating model...")
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc='Evaluating'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update accuracy
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        
        # Calculate metrics
        accuracy_score = total_correct / total_samples * 100
        
        # Generate classification report
        class_report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Save results
        self._save_classification_report(class_report, save_dir)
        self._plot_confusion_matrix(cm, save_dir)
        self._plot_class_accuracies(class_report, save_dir)
        
        # Calculate per-class metrics
        results = {
            'overall_accuracy': accuracy_score,
            'macro_avg_precision': class_report['macro avg']['precision'] * 100,
            'macro_avg_recall': class_report['macro avg']['recall'] * 100,
            'macro_avg_f1': class_report['macro avg']['f1-score'] * 100,
            'weighted_avg_precision': class_report['weighted avg']['precision'] * 100,
            'weighted_avg_recall': class_report['weighted avg']['recall'] * 100,
            'weighted_avg_f1': class_report['weighted avg']['f1-score'] * 100
        }
        
        print(f"\nEvaluation Results:")
        print(f"Overall Accuracy: {accuracy_score:.2f}%")
        print(f"Macro Avg F1-Score: {results['macro_avg_f1']:.2f}%")
        print(f"Weighted Avg F1-Score: {results['weighted_avg_f1']:.2f}%")
        
        return results
    
    def _save_classification_report(self, report: Dict, save_dir: str):
        """Save classification report to file."""
        report_path = os.path.join(save_dir, 'classification_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Per-class metrics
            for class_name in self.class_names:
                if class_name in report:
                    metrics = report[class_name]
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {metrics['support']}\n\n")
            
            # Overall metrics
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report:
                    metrics = report[avg_type]
                    f.write(f"{avg_type.title()}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n\n")
        
        print(f"Classification report saved to: {report_path}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_dir: str):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {cm_path}")
    
    def _plot_class_accuracies(self, report: Dict, save_dir: str):
        """Plot per-class accuracies."""
        class_metrics = []
        
        for class_name in self.class_names:
            if class_name in report:
                class_metrics.append({
                    'class': class_name,
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score']
                })
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['precision', 'recall', 'f1_score']
        metric_names = ['Precision', 'Recall', 'F1-Score']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [m[metric] for m in class_metrics]
            class_names_short = [c[:10] + '...' if len(c) > 10 else c for c in self.class_names]
            
            bars = axes[i].bar(class_names_short, values, color=f'C{i}', alpha=0.7)
            axes[i].set_title(f'Per-Class {name}', fontweight='bold')
            axes[i].set_ylabel(name)
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.01,
                    f'{value:.3f}', 
                    ha='center', 
                    va='bottom',
                    fontsize=8
                )
        
        plt.tight_layout()
        
        # Save plot
        metrics_path = os.path.join(save_dir, 'class_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class metrics plot saved to: {metrics_path}")


def compare_student_teacher(
    student_model: ViTForFishClassification,
    teacher_model: ViTForFishClassification,
    data_loader: DataLoader,
    class_names: List[str],
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Compare student and teacher model performance.
    
    Args:
        student_model: Student model
        teacher_model: Teacher model
        data_loader: DataLoader for evaluation
        class_names: List of class names
        device: Device for evaluation
        
    Returns:
        Dictionary with student and teacher results
    """
    print("Comparing Student and Teacher Models...")
    
    # Evaluate student
    print("\nEvaluating Student Model:")
    student_evaluator = ModelEvaluator(student_model, class_names, device)
    student_results = student_evaluator.evaluate_dataset(
        data_loader, 
        save_dir='./evaluation_results/student'
    )
    
    # Evaluate teacher
    print("\nEvaluating Teacher Model:")
    teacher_evaluator = ModelEvaluator(teacher_model, class_names, device)
    teacher_results = teacher_evaluator.evaluate_dataset(
        data_loader, 
        save_dir='./evaluation_results/teacher'
    )
    
    # Print comparison
    print(f"\n{'='*60}")
    print("STUDENT vs TEACHER COMPARISON")
    print(f"{'='*60}")
    
    for metric in ['overall_accuracy', 'macro_avg_f1', 'weighted_avg_f1']:
        student_val = student_results[metric]
        teacher_val = teacher_results[metric]
        improvement = teacher_val - student_val
        
        print(f"{metric.replace('_', ' ').title()}:")
        print(f"  Student: {student_val:.2f}%")
        print(f"  Teacher: {teacher_val:.2f}%")
        print(f"  Improvement: {improvement:+.2f}%")
        print()
    
    return {
        'student': student_results,
        'teacher': teacher_results
    }


def main():
    """Example evaluation script."""
    # Configuration
    checkpoint_path = './checkpoints/model_best.pth'
    data_dir = '/path/to/fish/dataset'  # Update this
    device = get_device()
    
    # Load data
    _, val_loader, class_names = create_fish_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        image_size=224
    )
    
    # Create models
    num_classes = len(class_names)
    student_model = ViTForFishClassification(num_classes=num_classes)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path, student_model)
        
        # Create teacher model and load weights
        teacher_model = ViTForFishClassification(num_classes=num_classes)
        teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
        
        # Compare models
        results = compare_student_teacher(
            student_model, teacher_model, val_loader, class_names, device
        )
        
        print("Evaluation completed! Check ./evaluation_results/ for detailed outputs.")
    
    else:
        print(f"Checkpoint not found: {checkpoint_path}")


if __name__ == '__main__':
    main()
