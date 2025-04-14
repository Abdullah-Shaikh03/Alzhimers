import os
import torch
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    average_precision_score, confusion_matrix,
    accuracy_score, f1_score, recall_score, precision_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from DataLoader import get_dataloaders
from Model import get_model
from Losses import anomaly_score


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze anomaly detection model performance")
    parser.add_argument('--config', type=str, default='Config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='Results', help='Directory to save analysis results')
    parser.add_argument('--normal_class', type=str, default='CN', help='Class to treat as normal (CN)')
    return parser.parse_args()


def load_model(model_path, config):
    """Load trained model from checkpoint"""
    device = torch.device(config['device'])
    model = get_model(config)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def get_anomaly_scores(model, dataloader, normal_class_idx, config):
    """Calculate anomaly scores for all samples in the dataloader"""
    device = torch.device(config['device'])
    scores = []
    labels = []
    image_paths = []
    original_labels = []
    predictions = []
    
    print("Calculating anomaly scores...")
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data = data.to(device)
            batch_size = data.size(0)
            
            # Get reconstructions based on model type
            if config['model_type'].lower() == 'vae':
                recon_batch, _, _ = model(data)
                batch_scores = torch.mean((recon_batch - data) ** 2, dim=[1, 2, 3]).cpu().numpy()
            else:  # gan
                encoded_z = model.encode(data)
                recon_batch = model.generator(encoded_z)
                batch_scores = torch.mean((recon_batch - data) ** 2, dim=[1, 2, 3]).cpu().numpy()
            
            # Binary labels: 0 for normal, 1 for anomaly
            binary_labels = (label != normal_class_idx).cpu().numpy().astype(int)
            
            scores.extend(batch_scores)
            labels.extend(binary_labels)
            original_labels.extend(label.cpu().numpy())
            
            # Calculate predictions based on threshold
            batch_preds = (batch_scores > config['anomaly_threshold']).astype(int)
            predictions.extend(batch_preds)
    
    return np.array(scores), np.array(labels), np.array(original_labels), np.array(predictions)


def calculate_metrics(scores, labels, threshold=None):
    """Calculate classification metrics based on anomaly scores"""
    if threshold is None:
        # Find optimal threshold based on ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    
    # Calculate predictions using threshold
    preds = (scores >= threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    
    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'roc_fpr': fpr,
        'roc_tpr': tpr,
        'roc_auc': roc_auc,
        'pr_precision': precision_curve,
        'pr_recall': recall_curve,
        'pr_auc': pr_auc,
        'threshold': threshold
    }


def analyze_class_performance(scores, original_labels, classes_dict, threshold=None):
    """Analyze model performance for each class"""
    class_metrics = {}
    
    # For each class, treat it as anomaly and all others as normal
    for class_name, class_idx in classes_dict.items():
        class_labels = (original_labels == class_idx).astype(int)
        
        if threshold is None:
            # Find optimal threshold for this class
            fpr, tpr, thresholds = roc_curve(class_labels, scores)
            optimal_idx = np.argmax(tpr - fpr)
            class_threshold = thresholds[optimal_idx]
        else:
            class_threshold = threshold
        
        # Calculate metrics for this class
        preds = (scores >= class_threshold).astype(int)
        
        acc = accuracy_score(class_labels, preds)
        precision = precision_score(class_labels, preds, zero_division=0)
        recall = recall_score(class_labels, preds)
        f1 = f1_score(class_labels, preds)
        
        fpr, tpr, _ = roc_curve(class_labels, scores)
        roc_auc = auc(fpr, tpr)
        
        class_metrics[class_name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'threshold': class_threshold
        }
    
    return class_metrics


def save_results(metrics, class_metrics, config, output_dir):
    """Save analysis results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall metrics
    results = {
        'model_type': config['model_type'],
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'threshold': metrics['threshold']
    }
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_dir / 'metrics.csv', index=False)
    
    # Save confusion matrix
    np.save(output_dir / 'confusion_matrix.npy', metrics['confusion_matrix'])
    
    # Save class metrics
    class_results = []
    for class_name, class_metric in class_metrics.items():
        class_results.append({
            'class': class_name,
            'accuracy': class_metric['accuracy'],
            'precision': class_metric['precision'],
            'recall': class_metric['recall'],
            'f1_score': class_metric['f1_score'],
            'roc_auc': class_metric['roc_auc'],
            'threshold': class_metric['threshold']
        })
    
    class_df = pd.DataFrame(class_results)
    class_df.to_csv(output_dir / 'class_metrics.csv', index=False)
    
    # Save ROC and PR curve data
    np.savez(output_dir / 'roc_data.npz', 
             fpr=metrics['roc_fpr'], 
             tpr=metrics['roc_tpr'],
             auc=metrics['roc_auc'])
    
    np.savez(output_dir / 'pr_data.npz',
             precision=metrics['pr_precision'],
             recall=metrics['pr_recall'],
             auc=metrics['pr_auc'])
    
    print(f"Results saved to {output_dir}")
    return results_df, class_df


def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data loaders
    _, _, test_loader = get_dataloaders(config)
    
    # Get class to idx mapping from test dataset
    class_to_idx = test_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    normal_class_idx = class_to_idx[args.normal_class]
    
    # Load model
    model = load_model(args.model_path, config)
    
    # Get anomaly scores
    scores, labels, original_labels, predictions = get_anomaly_scores(
        model, test_loader, normal_class_idx, config
    )
    
    # Calculate metrics
    metrics = calculate_metrics(scores, labels)
    print(f"Overall metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR AUC: {metrics['pr_auc']:.4f}")
    print(f"  Optimal threshold: {metrics['threshold']:.4f}")
    
    # Calculate class-specific metrics
    class_metrics = analyze_class_performance(scores, original_labels, class_to_idx)
    
    # Print class metrics
    for class_name, class_metric in class_metrics.items():
        print(f"\nClass {class_name} metrics:")
        print(f"  Accuracy: {class_metric['accuracy']:.4f}")
        print(f"  Precision: {class_metric['precision']:.4f}")
        print(f"  Recall: {class_metric['recall']:.4f}")
        print(f"  F1 Score: {class_metric['f1_score']:.4f}")
        print(f"  ROC AUC: {class_metric['roc_auc']:.4f}")
    
    # Save results
    save_results(metrics, class_metrics, config, args.output_dir)


if __name__ == "__main__":
    main()