import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from .mlize import models
from .disk_storage import load_all_results
import polars as pl  # se agrega para trabajar con polars
from matplotlib import cm
import matplotlib.colors as mcolors

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
    # Convierte el diccionario de métricas en una DataFrame usando polars
    df = pl.DataFrame([{"Model": k, **v} for k, v in results_dict.items()])
    return df

def plot_grouped_bar(results_df, vectorization_name, output_dir):
    # Realiza el melt usando Polars (queda como Polars DataFrame)
    results_melted = results_df.melt(
        id_vars='Model',
        variable_name='Metric',
        value_name='Score'
    ).filter(~pl.col("Metric").is_in(["Correct Predictions", "Incorrect Predictions", "K-mer"]))
    
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
            alpha = max(0.3, score)  # Minimum alpha of 0.3 to ensure visibility
            model_colors_with_alpha.append((r, g, b, alpha))
        
        plt.bar(x + i * bar_width, model_scores, width=bar_width, label=model, color=model_colors_with_alpha)

    plt.title(f'{vectorization_name} Comparison of Metrics Across Models', fontsize=18)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(x + bar_width * (len(unique_models) - 1) / 2, unique_metrics, rotation=45, fontsize=12)
    plt.ylim(0, 1)
    plt.legend(title='Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{vectorization_name}_grouped_bar.png")
    plt.close()

def plot_correct_incorrect_bar(results_df, vectorization_name, output_dir):
    """
    Genera un gráfico de barras comparando las predicciones correctas e incorrectas,
    extrayendo los datos directamente de un DataFrame de Polars.

    Parameters:
        results_df (pl.DataFrame): DataFrame de Polars con columnas 'Model',
            'Correct Predictions' e 'Incorrect Predictions'.
        vectorization_name (str): Nombre de la técnica de vectorización (para el título del gráfico).
        output_dir (str): Directorio donde se guardará el gráfico.
    """
    # Realiza el melt con Polars
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
            
            # Normalize alpha based on prediction type
            if p_type == "Correct Predictions":
                # For correct predictions, higher is better
                max_correct = np.max(counts[pred_types == "Correct Predictions"]) if np.any(pred_types == "Correct Predictions") else 1
                alpha = max(0.3, count_val / max_correct if max_correct > 0 else 0.3)
            else:
                # For incorrect predictions, lower is better (invert normalization)
                max_incorrect = np.max(counts[pred_types == "Incorrect Predictions"]) if np.any(pred_types == "Incorrect Predictions") else 1
                alpha = max(0.3, 1 - (count_val / max_incorrect if max_incorrect > 0 else 0))
            
            model_colors_with_alpha.append((r, g, b, alpha))
        
        plt.bar(x + i * bar_width, model_counts, width=bar_width, label=model, color=model_colors_with_alpha)
    
    plt.title(f"{vectorization_name} Correct vs. Incorrect Classifications", fontsize=14)
    plt.xlabel("Prediction Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(x + bar_width * (len(unique_models) - 1) / 2, unique_pred_types)
    plt.legend(title="Model", loc="upper right")
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/{vectorization_name}_correct_incorrect_bar.png")
    plt.close()


def plot_roc_curve_by_model(X_test, y_test, output_dir):
    """
    Plots ROC curves for multiple models and their respective classes, and displays the average AUC for each model.
    """
    classes = sorted(y_test.unique())
    y_test_binarized = label_binarize(y_test, classes=classes)
    model_aucs = {}
    
    # Get the model color dictionary
    model_colors = get_model_colors()

    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            aucs = []
            class_aucs = []
            for i, class_name in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                class_aucs.append((class_name, fpr, tpr, roc_auc))
            
            avg_auc = np.mean(aucs)
            model_aucs[model_name] = (avg_auc, class_aucs, y_score)
        else:
            print(f"{model_name} does not support predict_proba and will be skipped.")
    
    sorted_models = sorted(model_aucs.items(), key=lambda x: x[1][0], reverse=True)
    for model_name, (avg_auc, class_aucs, y_score) in sorted_models:
        sorted_classes = sorted(class_aucs, key=lambda x: x[3], reverse=True)
        
        plt.figure(figsize=(8, 6))
        
        # Get the model's assigned color
        model_color = model_colors.get(model_name, '#333333')
        
        # Use different line styles for different classes but keep the same color for the model
        line_styles = ['-', '--', '-.', ':']
        
        for i, (class_name, fpr, tpr, roc_auc) in enumerate(sorted_classes):
            # Cycle through line styles if there are more classes than styles
            line_style = line_styles[i % len(line_styles)]
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})", 
                     color=model_color, linestyle=line_style, linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title(f"ROC Curve: {model_name} (Avg AUC = {avg_auc:.2f})", fontsize=16)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid()
        # Guardar cada ROC curve en la carpeta output con un nombre basado en el modelo
        safe_model_name = model_name.replace(" ", "_")
        plt.savefig(f"{output_dir}/roc_curve_{safe_model_name}.png")
        plt.close()

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
            alpha = max(0.3, (value + 1) / 2)
        else:
            alpha = max(0.3, value)
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
    # Fix: Use a generic name for the output file instead of vectorization_name
    plt.savefig(f"{output_dir}/bacc_mcc_4panels.png")
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
    
    data = []
    for method in all_results:
        for vectorization, model_metrics in method.items():
            for model_name, metrics in model_metrics.items():
                row = {
                    "Vectorization": vectorization,
                    "Model": model_name
                }
                row.update(metrics)
                data.append(row)
    return pl.DataFrame(data)

def best_k_accuracy():
    # Load results from disk
    all_results = load_all_results()
    
    if not all_results:
        return None

    k_accuracy = {}
    
    for item in all_results:
        for vectorization_method, models in item.items():
            for model_name, metrics in models.items():
                k_val = metrics["K-mer"]
                accuracy = metrics["Accuracy"]
                
                if k_val not in k_accuracy:
                    k_accuracy[k_val] = {"total": 0.0, "count": 0}
                
                k_accuracy[k_val]["total"] += accuracy
                k_accuracy[k_val]["count"] += 1

    # Se calcula el promedio y se selecciona el k con mayor accuracy promedio.
    best_k = max(
        k_accuracy.items(),
        key=lambda x: x[1]["total"] / x[1]["count"] if x[1]["count"] > 0 else float('-inf')
    )[0]
    
    return best_k

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