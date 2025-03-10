import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from .mlize import models
from .disk_storage import load_all_results, ensure_output_subdirs
import polars as pl  # se agrega para trabajar con polars
from matplotlib import cm
import matplotlib.colors as mcolors
from .logging import log_step  # Add import for log_step function
import os  # Add import for os module

# Define the model-specific color palette
def get_model_colors():
    """Return a dictionary mapping model names to specific colors"""
    return {
        'Random Forest': '#617EB6',          # Blue
        'Logistic Regression': '#37BA9A',    # Teal green
        'Support Vector Machine (RBF Kernel)': '#B772AF'  # Purple
    }

# Keep the original colormap for metric-based coloring
def get_cmap_norm():
    """Return the standard colormap and normalizer for all plots"""
    cmap = cm.get_cmap('RdYlBu')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    return cmap, norm

def prepare_plot_df(results_dict):
    # Convert the metrics dictionary into a DataFrame using polars
    df = pl.DataFrame([{"Model": k, **v} for k, v in results_dict.items()])
    return df

def plot_grouped_bar(results_df, vectorization_name, output_dir):
    # Ensure plots directory exists
    ensure_output_subdirs(output_dir)
    plots_dir = os.path.join(output_dir, 'plots', 'bars')
    
    # Perform melt using Polars (remains as Polars DataFrame)
    results_melted = results_df.melt(
        id_vars='Model',
        variable_name='Metric',
        value_name='Score'
    ).filter(pl.col("Metric").is_in(["Accuracy", "AUC", "F1 Score"]))
    
    # Convertir cada columna a array de NumPy
    models = results_melted["Model"].to_numpy()
    metrics = results_melted["Metric"].to_numpy()
    scores = results_melted["Score"].to_numpy()

    # Obtener los valores únicos en cada dimensión
    unique_models = np.unique(models)
    unique_metrics = np.unique(metrics)
    
    # Get the model color dictionary
    model_colors = get_model_colors()
    
    # Configurar el gráfico
    x = np.arange(len(unique_metrics))
    bar_width = 0.8 / len(unique_models)

    plt.figure(figsize=(12, 8))
    
    # Dibujar cada grupo de barras para cada modelo
    for i, model in enumerate(unique_models):
        # Para cada métrica, extraer la puntuación correspondiente
        model_scores = []
        model_colors_with_alpha = []
        for metric in unique_metrics:
            # Buscamos la puntuación; asumimos que hay un único valor por (model, metric)
            mask = (models == model) & (metrics == metric)
            score = scores[mask][0] if np.any(mask) else 0
            model_scores.append(score)
            
            # Get base color and add alpha based on the score
            base_color = model_colors.get(model, '#333333')
            # Convert hex to RGBA and set alpha based on score
            r, g, b = tuple(int(base_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
            # Ensure alpha is within 0-1 range
            alpha = max(0.3, min(1.0, score))  # Minimum alpha of 0.3, maximum of 1.0
            model_colors_with_alpha.append((r, g, b, alpha))
        
        plt.bar(x + i * bar_width, model_scores, width=bar_width, label=model, color=model_colors_with_alpha)

    plt.title(f'{vectorization_name} Comparison of Metrics Across Models', fontsize=18)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(x + bar_width * (len(unique_models) - 1) / 2, unique_metrics, rotation=45, fontsize=12)
    plt.ylim(0, 1)
    plt.legend(title='Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{vectorization_name}_grouped_bar.png")
    plt.close()

def plot_correct_incorrect_bar(results_df, vectorization_name, output_dir):
    """
    Generates a bar chart comparing correct and incorrect predictions,
    extracting data directly from a Polars DataFrame.

    Parameters:
        results_df (pl.DataFrame): Polars DataFrame with columns 'Model',
            'Correct Predictions' and 'Incorrect Predictions'.
        vectorization_name (str): Name of the vectorization technique (for the chart title).
        output_dir (str): Directory where the chart will be saved.
    """
    # Ensure plots directory exists
    ensure_output_subdirs(output_dir)
    plots_dir = os.path.join(output_dir, 'plots', 'bars')
    
    # Perform melt with Polars
    df_melted = results_df.melt(
        id_vars="Model",
        value_vars=["Correct Predictions", "Incorrect Predictions"],
        variable_name="Prediction Type",
        value_name="Count"
    )
    
    # Extraer los datos a arrays de NumPy
    models = df_melted["Model"].to_numpy()
    pred_types = df_melted["Prediction Type"].to_numpy()
    counts = df_melted["Count"].to_numpy()
    
    # Obtener los valores únicos en cada dimensión
    unique_models = np.unique(models)
    unique_pred_types = np.unique(pred_types)
    
    # Get the model color dictionary
    model_colors = get_model_colors()
    
    # Configurar posiciones para las barras
    x = np.arange(len(unique_pred_types))
    bar_width = 0.8 / len(unique_models)
    
    plt.figure(figsize=(8, 5))
    
    # Dibujar barras para cada modelo
    for i, model in enumerate(unique_models):
        model_counts = []
        model_colors_with_alpha = []
        for p_type in unique_pred_types:
            mask = (models == model) & (pred_types == p_type)
            # Asumimos que hay un único valor para cada combinación
            count_val = counts[mask][0] if np.any(mask) else 0
            model_counts.append(count_val)
            
            # Get the base color and add alpha
            base_color = model_colors.get(model, '#333333')
            r, g, b = tuple(int(base_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
            
            # Normalize alpha based on prediction type and ensure it's within 0-1 range
            if p_type == "Correct Predictions":
                # For correct predictions, higher is better
                max_correct = np.max(counts[pred_types == "Correct Predictions"]) if np.any(pred_types == "Correct Predictions") else 1
                alpha = max(0.3, min(1.0, count_val / max_correct if max_correct > 0 else 0.3))
            else:
                # For incorrect predictions, lower is better (invert normalization)
                max_incorrect = np.max(counts[pred_types == "Incorrect Predictions"]) if np.any(pred_types == "Incorrect Predictions") else 1
                alpha = max(0.3, min(1.0, 1 - (count_val / max_incorrect if max_incorrect > 0 else 0)))
            
            model_colors_with_alpha.append((r, g, b, alpha))
        
        plt.bar(x + i * bar_width, model_counts, width=bar_width, label=model, color=model_colors_with_alpha)
    
    plt.title(f"{vectorization_name} Correct vs. Incorrect Classifications", fontsize=14)
    plt.xlabel("Prediction Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(x + bar_width * (len(unique_models) - 1) / 2, unique_pred_types)
    plt.legend(title="Model", loc="upper right")
    plt.tight_layout()
    
    plt.savefig(f"{plots_dir}/{vectorization_name}_correct_incorrect_bar.png")
    plt.close()


def plot_roc_curve_by_model(X_test, y_test, model, model_name, vectorization_name, output_dir):
    """
    Plots ROC curves for a model with proper micro-averaging and per-class curves.
    
    Parameters:
        X_test: Test features
        y_test: True labels
        model: The trained model instance
        model_name: Name of the model
        vectorization_name: Name of the vectorization method
        output_dir: Directory to save the plot
    """
    # Ensure plots directory exists
    ensure_output_subdirs(output_dir)
    plots_dir = os.path.join(output_dir, 'plots', 'roc')
    
    # Get the model color dictionary
    model_colors = get_model_colors()
    model_color = model_colors.get(model_name, '#333333')
    
    if not hasattr(model, "predict_proba"):
        log_step(f"{model_name} does not support predict_proba and will be skipped.")
        return
    
    # Get unique classes
    classes = np.unique(y_test)
    n_classes = len(classes)
    
    # Binarize the labels for ROC curve calculation
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Get prediction probabilities
    y_score = model.predict_proba(X_test)
    
    # Calculate ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Increase figure size to accommodate all elements - make it larger than before
    plt.figure(figsize=(15, 12))  # Increased size
    
    # Plot ROC for each class
    class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # First calculate and store all the values
    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot individual class ROC curves
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.7, color=class_colors[i],
                 label=f'ROC class {cls} (AUC = {roc_auc[i]:.2f})')
    
    # Compute macro-average ROC curve and ROC area
    macro_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    macro_tpr = np.zeros_like(macro_fpr)
    for i in range(n_classes):
        macro_tpr += np.interp(macro_fpr, fpr[i], tpr[i])
    
    # Average TPR values and compute AUC
    macro_tpr /= n_classes
    macro_auc = auc(macro_fpr, macro_tpr)
    
    # Plot macro-average ROC curve
    plt.plot(macro_fpr, macro_tpr, lw=2, color='navy',
             label=f'Macro-average ROC (AUC = {macro_auc:.2f})',
             linestyle='-')
    
    # Compute micro-average ROC curve and ROC area
    y_test_flat = y_test_bin.ravel()
    y_score_flat = np.concatenate([y_score[:, i] for i in range(n_classes)])
    
    # Debug: Log data shapes to verify correct processing
    log_step(f"Shape of flattened test data: {y_test_flat.shape}")
    log_step(f"Shape of flattened score data: {y_score_flat.shape}")
    
    micro_fpr, micro_tpr, _ = roc_curve(y_test_flat, y_score_flat)
    micro_auc = auc(micro_fpr, micro_tpr)
    
    # Plot micro-average ROC curve with the model's color
    plt.plot(micro_fpr, micro_tpr, lw=3, color=model_color,
             label=f'Micro-average ROC (AUC = {micro_auc:.2f})',
             linestyle='-')
    
    # Plot the random guessing line
    plt.plot([0, 1], [0, 1], 'k--', label='Random guessing', alpha=0.8)
    
    # Add plot details
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f"ROC Curves: {model_name}\n{vectorization_name}", fontsize=16)
    
    # Adjust legend position and font size based on number of classes
    if n_classes > 10:
        # For many classes, move legend outside the plot
        plt.legend(loc="center left", fontsize=8, bbox_to_anchor=(1.02, 0.5))
    else:
        plt.legend(loc="lower right", fontsize=10)
        
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the figure with a more descriptive name
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    safe_vectorization = vectorization_name.replace(" ", "_")
    
    # Use tight_layout instead of constrained_layout
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/roc_curve_{safe_vectorization}_{safe_model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log the AUC values for reference
    log_step(f"ROC AUC for {model_name} ({vectorization_name}):")
    log_step(f"  Micro-average AUC: {micro_auc:.4f}")
    log_step(f"  Macro-average AUC: {macro_auc:.4f}")
    # for i, cls in enumerate(classes):
    #     log_step(f"  Class '{cls}' AUC: {roc_auc[i]:.4f}")

# Keep the original function with different name for backward compatibility
def plot_roc_curves_from_models(X_test, y_test, output_dir):
    """Legacy function - kept for backward compatibility"""
    from .logging import log_step
    log_step("Warning: Using deprecated ROC curve function. Consider updating code.")
    
    # Attempt to import models but warn this is not recommended
    from .mlize import models
    
    classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)
    
    # Get the model color dictionary
    model_colors = get_model_colors()
    
    for model_name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                log_step(f"Attempting to generate ROC curve using global model template: {model_name}")
                # ...rest of the implementation similar to original
                # This function should generally not be used
        except Exception as e:
            log_step(f"Error in deprecated ROC curve function: {str(e)}")

def plot_bacc_mcc_4panels(results_df, output_dir):
    """
    Lee un TSV con columnas:
      - Classifier
      - BACC_CV, MCC_CV
      - BACC_Test, MCC_Test
    y genera 4 subplots (2x2) como en la figura adjunta:
      (a) BACC (CV)
      (b) MCC (CV)
      (c) BACC (Test)
      (d) MCC (Test)
    con degradado de color de 0 a 1 en cada barra.
    """
    # For global_plots directory, ensure the bars subdirectory exists
    if output_dir.endswith('global_plots'):
        plots_dir = os.path.join(output_dir, 'bars')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
    else:
        # Regular k-mer subdirectory
        ensure_output_subdirs(output_dir)
        plots_dir = os.path.join(output_dir, 'plots', 'bars')
    
    # 1) Get model colors
    model_colors = get_model_colors()
    
    # 2) Extraer columnas
    classifiers = results_df["Model"].to_list()
    bacc_cv     = results_df["BACC_CV"].to_list()
    mcc_cv      = results_df["MCC_CV"].to_list()
    bacc_test   = results_df["BACC_Test"].to_list()
    mcc_test    = results_df["MCC_Test"].to_list()

    # Create colors with alpha based on values
    def get_color_with_alpha(base_color, value, normalize=True):
        r, g, b = tuple(int(base_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
        # For MCC, normalize from [-1,1] to [0,1]
        if normalize and value < 0:
            alpha = max(0.3, min(1.0, (value + 1) / 2))
        else:
            alpha = max(0.3, min(1.0, value))  # Ensure alpha is within 0-1 range
        return (r, g, b, alpha)

    bacc_cv_colors = [get_color_with_alpha(model_colors.get(model, '#333333'), val) 
                     for model, val in zip(classifiers, bacc_cv)]
    
    bacc_test_colors = [get_color_with_alpha(model_colors.get(model, '#333333'), val) 
                       for model, val in zip(classifiers, bacc_test)]
    
    mcc_cv_colors = [get_color_with_alpha(model_colors.get(model, '#333333'), val, normalize=True) 
                    for model, val in zip(classifiers, mcc_cv)]
    
    mcc_test_colors = [get_color_with_alpha(model_colors.get(model, '#333333'), val, normalize=True) 
                      for model, val in zip(classifiers, mcc_test)]
    
    # 3) Eje Y (posiciones)
    y_pos = np.arange(len(classifiers))

    # 4) Crear la figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # -- Subplot (a) BACC (CV) --
    axes[0, 0].barh(y_pos, bacc_cv, color=bacc_cv_colors)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(classifiers)
    axes[0, 0].invert_yaxis()  # El primer clasificador arriba
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_xlabel('BACC (CV)')
    axes[0, 0].set_title('(a)')

    # -- Subplot (b) MCC (CV) --
    axes[0, 1].barh(y_pos, mcc_cv, color=mcc_cv_colors)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(classifiers)
    axes[0, 1].invert_yaxis()
    # Ajustar ejes si MCC está en [-1,1] o [0,1], depende de tus valores
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_xlabel('MCC (CV)')
    axes[0, 1].set_title('(b)')

    # -- Subplot (c) BACC (Test) --
    axes[1, 0].barh(y_pos, bacc_test, color=bacc_test_colors)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(classifiers)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_xlabel('BACC (Test)')
    axes[1, 0].set_title('(c)')

    # -- Subplot (d) MCC (Test) --
    axes[1, 1].barh(y_pos, mcc_test, color=mcc_test_colors)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(classifiers)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim([0, 1])  # Ajusta según el rango real de MCC
    axes[1, 1].set_xlabel('MCC (Test)')
    axes[1, 1].set_title('(d)')

    plt.tight_layout()
    # Save to the appropriate directory
    plt.savefig(f"{plots_dir}/bacc_mcc_4panels.png")
    plt.close()


def results_to_df():
    """
    Converts the results of various vectorization methods and models into a polars DataFrame.

    This function loads results from disk and consolidates them into a DataFrame where each row 
    represents a unique combination of vectorization method and model, along with their associated metrics.

    Returns:
        pl.DataFrame: A DataFrame where each row contains the vectorization method, model, and their 
                      corresponding metrics.
    """
    # Load results from disk
    all_results = load_all_results()
    from .logging import log_step
    
    if not all_results:
        log_step("No results found in disk storage.")
        return pl.DataFrame()
    
    data = []
    missing_k = 0
    
    for method in all_results:
        for vectorization, model_metrics in method.items():
            for model_name, metrics in model_metrics.items():
                row = {
                    "Vectorization": vectorization,
                    "Model": model_name
                }
                
                # Check if K-mer exists in metrics
                if "K-mer" not in metrics:
                    missing_k += 1
                    log_step(f"Missing K-mer in results for {vectorization} - {model_name}")
                
                row.update(metrics)
                data.append(row)
    
    if missing_k > 0:
        log_step(f"Warning: {missing_k} results missing K-mer information. This might affect best_k_accuracy calculations.")
    
    return pl.DataFrame(data)

def best_k_accuracy():
    """
    Finds the k-mer size with the highest average accuracy across all models and vectorizations.
    Handles cases where the 'K-mer' key might be missing in some results.
    
    Returns:
        int or None: The best k-mer size, or None if no valid results found
    """
    # Load results from disk
    all_results = load_all_results()
    
    if not all_results:
        return None

    from .logging import log_step
    
    k_accuracy = {}
    skipped = 0
    
    for item in all_results:
        for vectorization_method, models in item.items():
            for model_name, metrics in models.items():
                try:
                    # Check if K-mer exists in the metrics
                    if "K-mer" not in metrics:
                        skipped += 1
                        continue
                        
                    k_val = metrics["K-mer"]
                    
                    # Also check if Accuracy exists
                    if "Accuracy" not in metrics:
                        skipped += 1
                        continue
                        
                    accuracy = metrics["Accuracy"]
                    
                    if k_val not in k_accuracy:
                        k_accuracy[k_val] = {"total": 0.0, "count": 0}
                    
                    k_accuracy[k_val]["total"] += accuracy
                    k_accuracy[k_val]["count"] += 1
                except Exception as e:
                    skipped += 1
                    log_step(f"Warning: Skipped a result when determining best K. Error: {str(e)}")
    
    if skipped > 0:
        log_step(f"Warning: Skipped {skipped} results when determining best K.")
    
    if not k_accuracy:
        log_step("No valid K-mer results found to determine best K.")
        return None
        
    # Calculate average accuracy for each k value and select the best
    try:
        best_k = max(
            k_accuracy.items(),
            key=lambda x: x[1]["total"] / x[1]["count"] if x[1]["count"] > 0 else float('-inf')
        )[0]
        log_step(f"Best K-mer determined: {best_k}")
        return best_k
    except Exception as e:
        log_step(f"Error calculating best K: {str(e)}")
        # Return the first K if we can't determine the best one
        return next(iter(k_accuracy.keys())) if k_accuracy else None

def get_best_k():
    # Load results from disk
    all_results = load_all_results()
    
    if not all_results:
        return None
    metrics_keys = ["Accuracy", "F1 Score", "AUC", "MCC"]
    k_scores = {}
    
    for item in all_results:
        for vectorization_method, models in item.items():
            for model_name, metrics in models.items():
                k_val = metrics["K-mer"]
                current_score = sum(metrics[key] for key in metrics_keys)
                
                if k_val not in k_scores:
                    k_scores[k_val] = {"total": 0.0, "count": 0}
                
                k_scores[k_val]["total"] += current_score
                k_scores[k_val]["count"] += 1

    best_k = max(
        k_scores.items(),
        key=lambda x: x[1]["total"]/x[1]["count"]
    )[0]

    return best_k

def plot_confusion_matrices(y_true, y_pred, vectorization_name, model_name, output_dir):
    """
    Generate and save normalized confusion matrices for model predictions.
    
    Parameters:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        vectorization_name: Name of the vectorization method
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    # Ensure plots directory exists
    ensure_output_subdirs(output_dir)
    plots_dir = os.path.join(output_dir, 'plots', 'confusion_matrix')
    
    # Get unique class names
    class_names = np.unique(np.concatenate([y_true, y_pred]))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Normalize the confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
    
    # Plot normalized matrix (as decimal values between 0-1)
    plt.figure(figsize=(max(8, len(class_names)*0.5), max(6, len(class_names)*0.4)))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Normalized Confusion Matrix - {model_name}\n{vectorization_name}')
    plt.ylabel('True Species')
    plt.xlabel('Predicted Species')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the normalized figure
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    safe_vectorization = vectorization_name.replace(" ", "_")
    plt.savefig(f"{plots_dir}/confusion_matrix_normalized_{safe_vectorization}_{safe_model_name}.png")
    plt.close()

def plot_feature_importance_heatmap(model, feature_names, vectorization_name, model_name, output_dir):
    """
    Generate and save a heatmap of feature importances for the given model.
    
    Parameters:
        model: Trained model with feature_importances_ attribute or coef_ attribute
        feature_names: List of feature names
        vectorization_name: Name of the vectorization method
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    # Ensure plots directory exists
    ensure_output_subdirs(output_dir)
    
    importances = None
    
    # Get feature importances based on model type
    if hasattr(model, 'feature_importances_'):  # Random Forest
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):  # Linear models like Logistic Regression
        importances = np.abs(model.coef_).sum(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    
    if importances is not None:
        # Get the top N important features
        n_features = min(30, len(feature_names))  # Limit to top 30 features or less
        indices = np.argsort(importances)[-n_features:]
        top_features = [feature_names[i] for i in indices]
        top_importances = [importances[i] for i in indices]
        
        # Sort in descending order for better visualization
        top_features = [x for _, x in sorted(zip(top_importances, top_features), reverse=True)]
        top_importances = sorted(top_importances, reverse=True)
        
        # Create bar chart version in the 'bars' subdirectory
        plt.figure(figsize=(10, max(4, n_features * 0.3)))
        bars_dir = os.path.join(output_dir, 'plots', 'bars')
        
        # Create horizontal bar chart for better readability with many features
        plt.barh(range(len(top_features)), top_importances, color=get_model_colors().get(model_name, '#333333'))
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {n_features} Feature Importance - {model_name}\n{vectorization_name}')
        plt.tight_layout()
        
        # Save the figure
        safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        safe_vectorization = vectorization_name.replace(" ", "_")
        plt.savefig(f"{bars_dir}/feature_importance_{safe_vectorization}_{safe_model_name}.png")
        plt.close()
        
        # Create a heatmap version in the 'heatmap' subdirectory
        plt.figure(figsize=(12, max(4, n_features * 0.25)))
        heatmap_dir = os.path.join(output_dir, 'plots', 'heatmap')
        
        data = np.zeros((1, len(top_features)))
        for i, importance in enumerate(top_importances):
            data[0, i] = importance
            
        sns.heatmap(data, annot=True, fmt='.3f', cmap='viridis',
                    xticklabels=top_features, yticklabels=['Importance'])
        plt.title(f'Feature Importance Heatmap - {model_name}\n{vectorization_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the heatmap figure
        plt.savefig(f"{heatmap_dir}/feature_heatmap_{safe_vectorization}_{safe_model_name}.png")
        plt.close()

def plot_prediction_pie(y_true, y_pred, vectorization_name, model_name, output_dir):
    """
    Generate and save a pie chart showing the proportion of correct vs. incorrect predictions.
    
    Parameters:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        vectorization_name: Name of the vectorization method
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    # Ensure plots directory exists
    ensure_output_subdirs(output_dir)
    plots_dir = os.path.join(output_dir, 'plots', 'pie')
    
    # Calculate correct and incorrect predictions
    correct = np.sum(y_true == y_pred)
    incorrect = np.sum(y_true != y_pred)
    
    # Get per-class accuracy
    classes = np.unique(y_true)
    class_correct = {}
    class_total = {}
    
    for cls in classes:
        mask = y_true == cls
        class_total[cls] = np.sum(mask)
        class_correct[cls] = np.sum(np.logical_and(mask, y_true == y_pred))
    
    # Create overall pie chart
    labels = ['Correct', 'Incorrect']
    sizes = [correct, incorrect]
    colors = ['#66b3ff', '#ff9999']
    
    plt.figure(figsize=(10, 10))
    
    # First pie chart - overall accuracy
    plt.subplot(1, 2, 1)
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, shadow=True, explode=(0, 0.1))
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Overall Prediction Results\n{correct}/{correct+incorrect} Correct')
    
    # Second pie chart - per-class accuracy
    plt.subplot(1, 2, 2)
    class_labels = []
    class_sizes = []
    class_colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    explodes = [0.05] * len(classes)
    
    for i, cls in enumerate(classes):
        accuracy = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        class_labels.append(f"{cls}: {accuracy:.1%}")
        class_sizes.append(class_total[cls])
    
    plt.pie(class_sizes, labels=class_labels, colors=class_colors, 
            autopct=lambda p: f'{int(p*sum(class_sizes)/100)}', startangle=90, 
            shadow=True, explode=explodes)
    plt.axis('equal')
    plt.title('Class Distribution')
    
    plt.suptitle(f'Prediction Results - {model_name}\n{vectorization_name}', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    safe_vectorization = vectorization_name.replace(" ", "_")
    plt.savefig(f"{plots_dir}/prediction_pie_{safe_vectorization}_{safe_model_name}.png")
    plt.close()