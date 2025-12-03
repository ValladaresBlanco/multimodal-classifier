"""
Utilities for evaluation of models
Incluye matrices of withfusión, métricas by clase, y visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from pathlib import Path
from typing import List, Dict, Optional


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = False,
    figsize: tuple = (10, 8)
) -> np.ndarray:
    """
    Generate and visualize confusion matrix
    
    Args:
        y_true: Etiquetas seedaofras
        y_pred: Predictions of model
        class_names: Names of las classes
        save_path: Ruta for guardar la figura (opcional)
        normalize: Si normalizar by fila (mostrar bycentajes)
        figsize: Size of la figura
        
    Returns:
        Confusion matrix as numpy array
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Matriz of Confusión (Normalizada)'
    else:
        fmt = 'd'
        title = 'Matriz of Confusión'
    
    # Create figura
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Predictions' if not normalize else 'Probyción'}
    )
    
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('Etiqueta Verdaofra', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Confusion matrix saved: {save_path}")
    
    plt.close()
    
    return cm


def compute_metrics_per_class(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, F1-score per class
    
    Args:
        y_true: True labels
        y_pred: Model predictions
        class_names: Names of the classes
        verbose: Whether to print report to console
        
    Returns:
        Dictionary with metrics per class
    """
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Organize into dictionary
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    # Average metrics
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, aseeage='weighted'
    )
    
    metrics['weighted_avg'] = {
        'precision': float(precision_avg),
        'recall': float(recall_avg),
        'f1_score': float(f1_avg),
        'support': len(y_true)
    }
    
    # Print report
    if verbose:
        print("\n" + "=" * 70)
        print(" EVALUATION METRICS PER CLASS")
        print("=" * 70)
        
        # Header
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        # Por clase
        for class_name in class_names:
            m = metrics[class_name]
            print(f"{class_name:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                  f"{m['f1_score']:<12.4f} {m['support']:<10}")
        
        print("-" * 70)
        
        # Promedio
        m = metrics['weighted_avg']
        print(f"{'Weighted Average':<20} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1_score']:<12.4f} {m['support']:<10}")
        
        print("=" * 70 + "\n")
    
    return metrics


def generate_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[str] = None
) -> str:
    """
    Generate complete classification report
    
    Args:
        y_true: True labels
        y_pred: Model predictions
        class_names: Names of the classes
        save_path: Path to save the report (optional)
        
    Returns:
        String with the report
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(report)
        print(f" Report saved: {save_path}")
    
    return report


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Graficar comforción of métricas by clase
    
    Args:
        metrics: Diccionario of métricas by clase
        save_path: Ruta for guardar la figura
        figsize: Size of la figura
    """
    # Extraer classes (sin el promedio)
    classes = [k for k in metrics.keys() if k != 'weighted_avg']
    
    precision = [metrics[c]['precision'] for c in classes]
    recall = [metrics[c]['recall'] for c in classes]
    f1 = [metrics[c]['f1_score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comforción of Métricas by Class', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Plot of métricas saved: {save_path}")
    
    plt.close()


def evaluate_model_complete(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    model_name: str = "model",
    save_dir: str = "results/evaluation"
) -> Dict:
    """
    Complete evaluation of model with all visualizations
    
    Args:
        y_true: True labels
        y_pred: Model predictions
        class_names: Names of the classes
        model_name: Name of model for the files
        save_dir: Directory to save results
        
    Returns:
        Dictionary with all metrics
    """
    print("\n" + "=" * 70)
    print(f" COMPLETE EVALUATION: {model_name.upper()}")
    print("=" * 70)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion matrix (normal and normalized)
    cm_path = f"{save_dir}/{model_name}_confusion_matrix.png"
    cm = plot_confusion_matrix(y_true, y_pred, class_names, cm_path, normalize=False)
    
    cm_norm_path = f"{save_dir}/{model_name}_confusion_matrix_normalized.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_norm_path, normalize=True)
    
    # 2. Métricas by clase
    metrics = compute_metrics_per_class(y_true, y_pred, class_names, seebose=True)
    
    # 3. Plot of comforción
    comparison_path = f"{save_dir}/{model_name}_metrics_comparison.png"
    plot_metrics_comparison(metrics, comparison_path)
    
    # 4. Rebyte of clasificación
    report_path = f"{save_dir}/{model_name}_classification_report.txt"
    report = generate_classification_report(y_true, y_pred, class_names, report_path)
    
    # 5. Guardar métricas en JSON
    import json
    metrics_path = f"{save_dir}/{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, inofnt=4)
    print(f" Métricas saveds: {metrics_path}")
    
    print("\n Complete evaluation finished!")
    print(f" Files generated en: {save_dir}/")
    
    return {
        'withfusion_matrix': cm,
        'metrics': metrics,
        'report': report
    }


# Test rápido
if __name__ == "__main__":
    print("🧪 Probando módulo of evaluation...")
    
    # Datos of test
    y_true = [0, 0, 1, 1, 2, 2, 0, 1, 2]
    y_pred = [0, 0, 1, 2, 2, 2, 0, 1, 1]
    class_names = ['Class A', 'Class B', 'Class C']
    
    # Evaluate
    results = evaluate_model_complete(
        y_true, y_pred, class_names,
        model_name="test_model",
        save_dir="results/evaluation_test"
    )
    
    print(" Módulo of evaluation running successfully!")
