import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from .mlize import models
from . import all_results
import polars as pl  # se agrega para trabajar con polars

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
    
    # Configurar el gráfico
    x = np.arange(len(unique_metrics))
    bar_width = 0.8 / len(unique_models)

    plt.figure(figsize=(12, 8))
    
    # Dibujar cada grupo de barras para cada modelo
    for i, model in enumerate(unique_models):
        # Para cada métrica, extraer la puntuación correspondiente
        model_scores = []
        for metric in unique_metrics:
            # Buscamos la puntuación; asumimos que hay un único valor por (model, metric)
            mask = (models == model) & (metrics == metric)
            score = scores[mask][0] if np.any(mask) else 0
            model_scores.append(score)
        plt.bar(x + i * bar_width, model_scores, width=bar_width, label=model)

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
    
    # Configurar posiciones para las barras
    x = np.arange(len(unique_pred_types))
    bar_width = 0.8 / len(unique_models)
    
    plt.figure(figsize=(8, 5))
    
    # Dibujar barras para cada modelo
    for i, model in enumerate(unique_models):
        model_counts = []
        for p_type in unique_pred_types:
            mask = (models == model) & (pred_types == p_type)
            # Asumimos que hay un único valor para cada combinación
            count_val = counts[mask][0] if np.any(mask) else 0
            model_counts.append(count_val)
        plt.bar(x + i * bar_width, model_counts, width=bar_width, label=model)
    
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
    Parameters:
    X_test (array-like): Test features.
    y_test (array-like): True labels for the test set.
    Description:
    This function takes the test features and true labels, and for each model in the global `models` dictionary, it calculates the ROC curve and AUC for each class. It then plots the ROC curves for each class of each model, along with the average AUC for the model. Models that do not support the `predict_proba` method are skipped.
    The ROC curves are plotted with the false positive rate on the x-axis and the true positive rate on the y-axis. A diagonal line representing random guessing is also plotted for reference.
    The function displays the ROC curves using matplotlib, with each class's ROC curve labeled with its AUC, and the plot titled with the model name and its average AUC.
    Returns:
    None
    """
    classes = sorted(y_test.unique())
    y_test_binarized = label_binarize(y_test, classes=classes)
    model_aucs = {}

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
        for class_name, fpr, tpr, roc_auc in sorted_classes:
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
        
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

def results_to_df():
    """
    Converts the results of various vectorization methods and models into a polars DataFrame.

    This function iterates over a nested dictionary structure `all_results` where the first level 
    keys are vectorization methods, the second level keys are models, and the values are dictionaries 
    of metrics. It consolidates these results into a DataFrame where each row represents a unique 
    combination of vectorization method and model, along with their associated metrics.

    Returns:
        pl.DataFrame: A DataFrame where each row contains the vectorization method, model, and their 
                      corresponding metrics.
    """
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