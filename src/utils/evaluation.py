"""
Utilidades para evaluación de modelos
Incluye matrices de confusión, métricas por clase, y visualizaciones
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
    Generar y visualizar matriz de confusión
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        class_names: Nombres de las clases
        save_path: Ruta para guardar la figura (opcional)
        normalize: Si normalizar por fila (mostrar porcentajes)
        figsize: Tamaño de la figura
        
    Returns:
        Matriz de confusión como numpy array
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Matriz de Confusión (Normalizada)'
    else:
        fmt = 'd'
        title = 'Matriz de Confusión'
    
    # Crear figura
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Predicciones' if not normalize else 'Proporción'}
    )
    
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('Etiqueta Verdadera', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Matriz de confusión guardada: {save_path}")
    
    plt.close()
    
    return cm


def compute_metrics_per_class(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Calcular precision, recall, F1-score por clase
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        class_names: Nombres de las clases
        verbose: Si imprimir reporte en consola
        
    Returns:
        Diccionario con métricas por clase
    """
    # Calcular métricas
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Organizar en diccionario
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    # Métricas promedio
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    metrics['weighted_avg'] = {
        'precision': float(precision_avg),
        'recall': float(recall_avg),
        'f1_score': float(f1_avg),
        'support': len(y_true)
    }
    
    # Imprimir reporte
    if verbose:
        print("\n" + "=" * 70)
        print("📊 MÉTRICAS DE EVALUACIÓN POR CLASE")
        print("=" * 70)
        
        # Header
        print(f"{'Clase':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        # Por clase
        for class_name in class_names:
            m = metrics[class_name]
            print(f"{class_name:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                  f"{m['f1_score']:<12.4f} {m['support']:<10}")
        
        print("-" * 70)
        
        # Promedio
        m = metrics['weighted_avg']
        print(f"{'Promedio Ponderado':<20} {m['precision']:<12.4f} {m['recall']:<12.4f} "
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
    Generar reporte de clasificación completo
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        class_names: Nombres de las clases
        save_path: Ruta para guardar el reporte (opcional)
        
    Returns:
        String con el reporte
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE CLASIFICACIÓN\n")
            f.write("=" * 70 + "\n\n")
            f.write(report)
        print(f"📄 Reporte guardado: {save_path}")
    
    return report


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Graficar comparación de métricas por clase
    
    Args:
        metrics: Diccionario de métricas por clase
        save_path: Ruta para guardar la figura
        figsize: Tamaño de la figura
    """
    # Extraer clases (sin el promedio)
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
    
    ax.set_xlabel('Clase', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparación de Métricas por Clase', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Gráfica de métricas guardada: {save_path}")
    
    plt.close()


def evaluate_model_complete(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    model_name: str = "model",
    save_dir: str = "results/evaluation"
) -> Dict:
    """
    Evaluación completa del modelo con todas las visualizaciones
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        class_names: Nombres de las clases
        model_name: Nombre del modelo para los archivos
        save_dir: Directorio donde guardar resultados
        
    Returns:
        Diccionario con todas las métricas
    """
    print("\n" + "=" * 70)
    print(f"🎯 EVALUACIÓN COMPLETA: {model_name.upper()}")
    print("=" * 70)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Matriz de confusión (normal y normalizada)
    cm_path = f"{save_dir}/{model_name}_confusion_matrix.png"
    cm = plot_confusion_matrix(y_true, y_pred, class_names, cm_path, normalize=False)
    
    cm_norm_path = f"{save_dir}/{model_name}_confusion_matrix_normalized.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_norm_path, normalize=True)
    
    # 2. Métricas por clase
    metrics = compute_metrics_per_class(y_true, y_pred, class_names, verbose=True)
    
    # 3. Gráfica de comparación
    comparison_path = f"{save_dir}/{model_name}_metrics_comparison.png"
    plot_metrics_comparison(metrics, comparison_path)
    
    # 4. Reporte de clasificación
    report_path = f"{save_dir}/{model_name}_classification_report.txt"
    report = generate_classification_report(y_true, y_pred, class_names, report_path)
    
    # 5. Guardar métricas en JSON
    import json
    metrics_path = f"{save_dir}/{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Métricas guardadas: {metrics_path}")
    
    print("\n✅ Evaluación completa terminada!")
    print(f"📁 Archivos generados en: {save_dir}/")
    
    return {
        'confusion_matrix': cm,
        'metrics': metrics,
        'report': report
    }


# Test rápido
if __name__ == "__main__":
    print("🧪 Probando módulo de evaluación...")
    
    # Datos de prueba
    y_true = [0, 0, 1, 1, 2, 2, 0, 1, 2]
    y_pred = [0, 0, 1, 2, 2, 2, 0, 1, 1]
    class_names = ['Clase A', 'Clase B', 'Clase C']
    
    # Evaluar
    results = evaluate_model_complete(
        y_true, y_pred, class_names,
        model_name="test_model",
        save_dir="results/evaluation_test"
    )
    
    print("✅ Módulo de evaluación funciona correctamente!")
