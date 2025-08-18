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

from model import ViTForFishClassification
# from trainer import EMATeacher  # Not needed for evaluation
from data import create_dataloaders
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
        top5_correct = 0  # Add top-5 tracking
        
        print("Evaluating model...")
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc='Evaluating'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Calculate top-5 accuracy
                _, top5_pred = logits.topk(5, 1, True, True)
                top5_correct += (targets.view(-1, 1).expand_as(top5_pred) == top5_pred).sum().item()
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update accuracy
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        
        # Calculate metrics
        accuracy_score = total_correct / total_samples * 100
        top5_accuracy_score = top5_correct / total_samples * 100
        
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
            'accuracy': accuracy_score,  # Changed from 'overall_accuracy' to match main function
            'top5_accuracy': top5_accuracy_score,  # Add top-5 accuracy
            'macro_avg_precision': class_report['macro avg']['precision'] * 100,
            'macro_avg_recall': class_report['macro avg']['recall'] * 100,
            'macro_avg_f1': class_report['macro avg']['f1-score'] * 100,
            'weighted_avg_precision': class_report['weighted avg']['precision'] * 100,
            'weighted_avg_recall': class_report['weighted avg']['recall'] * 100,
            'weighted_avg_f1': class_report['weighted avg']['f1-score'] * 100,
            'classification_report': class_report  # Add classification report to results
        }
        
        print(f"\nEvaluation Results:")
        print(f"Top-1 Accuracy: {accuracy_score:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy_score:.2f}%")
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
    """Main evaluation script."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate ViT-FishID model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to fish dataset directory')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    
    args = parser.parse_args()
    
    print(f"🔍 Evaluating model: {args.model_path}")
    print(f"📊 Data directory: {args.data_dir}")
    
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    # Load data - use test_loader for evaluation
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    print(f"📊 Found {len(class_names)} classes: {class_names[:5]}..." if len(class_names) > 5 else f"📊 Found {len(class_names)} classes: {class_names}")
    
    # Create model
    num_classes = len(class_names)
    model = ViTForFishClassification(
        num_classes=num_classes,
        model_name='vit_small_patch16_224',  # Adjust based on your training config
        pretrained=False,
        dropout_rate=0.1
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(args.model_path):
        print(f"📥 Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'student_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['student_state_dict'])
            print(f"✅ Loaded student model weights")
            if 'best_accuracy' in checkpoint:
                print(f"📊 Training best accuracy: {checkpoint['best_accuracy']:.2f}%")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model weights")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights")
    else:
        print(f"❌ Checkpoint not found: {args.model_path}")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model, class_names, device)
    
    # Evaluate on test set
    print(f"\n🧪 Evaluating on test set...")
    test_results = evaluator.evaluate_dataset(test_loader, "test")
    
    # Print results
    print(f"\n📊 TEST RESULTS:")
    print(f"🎯 Accuracy: {test_results['accuracy']:.2f}%")
    print(f"📈 Top-5 Accuracy: {test_results.get('top5_accuracy', 'N/A')}")
    
    # Print per-class results
    if 'classification_report' in test_results:
        print(f"\n📋 Per-class Performance:")
        class_report = test_results['classification_report']
        for class_name in class_names[:10]:  # Show first 10 classes
            if class_name in class_report:
                precision = class_report[class_name]['precision']
                recall = class_report[class_name]['recall']
                f1 = class_report[class_name]['f1-score']
                print(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        if len(class_names) > 10:
            print(f"  ... and {len(class_names) - 10} more classes")
    
    # Also evaluate on validation set for comparison
    print(f"\n🔍 Evaluating on validation set...")
    val_results = evaluator.evaluate_dataset(val_loader, "validation")
    print(f"📊 VALIDATION RESULTS:")
    print(f"🎯 Accuracy: {val_results['accuracy']:.2f}%")
    
    print(f"\n✅ Evaluation completed!")
    print(f"📊 Final Test Accuracy: {test_results['accuracy']:.2f}%")


if __name__ == '__main__':
    main()
