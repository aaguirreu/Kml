import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from .mlize import models
from . import all_results

def prepare_plot_df(results_dict):
    # Converts a dictionary of metrics into a DataFrame, like the old transformations
    df = pd.DataFrame(results_dict).T.reset_index()
    df = df.rename(columns={'index': 'Model'})
    return df

def plot_grouped_bar(results_df, output_dir):
    """
    Plots a grouped bar chart comparing metrics across different models.

    Parameters:
    results_df (pd.DataFrame): A DataFrame containing the results with models as columns and metrics as rows.

    The function transforms the DataFrame, excluding 'Correct Predictions' and 'Incorrect Predictions' metrics,
    and then uses seaborn to create a grouped bar plot. The plot displays the comparison of various metrics
    across different models.

    The plot is customized with titles, labels, and a legend, and the metrics are normalized to a range of 0 to 1.

    Returns:
    None
    """
    results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    results_melted = results_melted[~results_melted['Metric'].isin(['Correct Predictions', 'Incorrect Predictions'])]

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=results_melted, x='Metric', y='Score', hue='Model', errorbar=None)
    plt.title('Comparison of Metrics Across Models', fontsize=18)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.ylim(0, 1)  # Normalizar las métricas
    plt.legend(title='Model', fontsize=12)
    plt.tight_layout()
    # Guardar el plot en lugar de mostrarlo
    plt.savefig(f"{output_dir}/grouped_bar.png")
    plt.close()

def plot_correct_incorrect_bar(results_df, output_dir):
    """
    Plots a bar chart comparing correct and incorrect predictions by vectorization method.

    This function takes a DataFrame containing prediction results for different vectorization methods
    and models, reshapes the data for visualization, and generates a bar chart that highlights the number
    of correct versus incorrect predictions. The resulting plot is saved to a specified output directory as
    a PNG file.

    Parameters:
        results_df (pandas.DataFrame): DataFrame with columns 'Vectorization', 'Model', 'Correct Predictions',
                                       and 'Incorrect Predictions', containing the data needed for the plot.
        output_dir (str): The path to the directory where the generated bar chart image will be saved.

    Returns:
        None
    """
    # Melt focusing on Correct vs. Incorrect
    df_melted = results_df.melt(
        id_vars="Model",
        value_vars=["Correct Predictions", "Incorrect Predictions"],
        var_name="Prediction Type",
        value_name="Count"
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_melted,
        x="Prediction Type",
        y="Count",
        hue="Model",
        errorbar=None
    )

    plt.title("Correct vs. Incorrect Predictions", fontsize=14)
    plt.xlabel("Prediction Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Model", loc="upper right")
    plt.tight_layout()

    plt.savefig(f"{output_dir}/correct_incorrect_bar.png")
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
    Converts the results of various vectorization methods and models into a pandas DataFrame.

    This function iterates over a nested dictionary structure `all_results` where the first level 
    keys are vectorization methods, the second level keys are models, and the values are dictionaries 
    of metrics. It consolidates these results into a DataFrame where each row represents a unique 
    combination of vectorization method and model, along with their associated metrics.

    Returns:
        pd.DataFrame: A DataFrame where each row contains the vectorization method, model, and their 
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
    return pd.DataFrame(data)