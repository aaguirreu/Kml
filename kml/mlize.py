from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import polars as pl
import numpy as np
from .logging import log_step
import re

# Define the models to evaluate.
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
}

def train_model(X_train, y_train, model):
    """Train the provided model using the training data."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, auc=True, cv=5):
    """
    Train and evaluate a model using cross-validation and a test set.

    Parameters:
        model: The machine learning model to evaluate.
        X_train, X_test: Feature sets for training and testing.
        y_train, y_test: Labels for training and testing.
        auc (bool): Whether to compute AUC (only valid if predict_proba is available).
        cv (int): Number of folds for cross-validation.

    Returns:
        dict: Dictionary containing evaluation metrics for both CV and test.
              - BACC_CV: Balanced Accuracy (CV average)
              - MCC_CV: Matthews Corr. Coef. (CV average)
              - BACC_Test: Balanced Accuracy (test set)
              - MCC_Test: Matthews Corr. Coef. (test set)
              - Accuracy, F1 Score, AUC, etc. (test set)
    """
    # --- 1) CROSS-VALIDATION ---
    bacc_scorer = make_scorer(balanced_accuracy_score)
    mcc_scorer  = make_scorer(matthews_corrcoef)

    # Promedio de Balanced Accuracy en CV
    bacc_cv = cross_val_score(model, X_train, y_train, scoring=bacc_scorer, cv=cv).mean()
    # Promedio de MCC en CV
    mcc_cv  = cross_val_score(model, X_train, y_train, scoring=mcc_scorer, cv=cv).mean()

    # --- 2) ENTRENAR MODELO FINAL en todo X_train ---
    model.fit(X_train, y_train)

    # --- 3) EVALUACIÓN en TEST ---
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Métricas en test
    accuracy_test = accuracy_score(y_test, y_pred)
    bacc_test     = balanced_accuracy_score(y_test, y_pred)
    f1_test       = f1_score(y_test, y_pred, average='weighted')
    mcc_test      = matthews_corrcoef(y_test, y_pred)

    # AUC en test (si procede)
    auc_score_test = None
    if auc and y_pred_proba is not None:
        if y_pred_proba.shape[1] == 2:
            # Para binario, usar la prob. de la clase positiva
            y_pred_proba = y_pred_proba[:, 1]
        auc_score_test = roc_auc_score(
            y_test,
            y_pred_proba,
            multi_class='ovr' if len(np.unique(y_test)) > 2 else None
        )

    correct_predictions    = np.sum(np.array(y_test) == np.array(y_pred))
    incorrect_predictions  = len(y_test) - correct_predictions

    # --- 4) RETORNAR RESULTADOS ---
    return {
        # -- CV Metrics --
        'BACC_CV': bacc_cv,
        'MCC_CV': mcc_cv,

        # -- Test Metrics --
        'BACC_Test': bacc_test,
        'MCC_Test': mcc_test,

        'Accuracy': accuracy_test,
        'F1 Score': f1_test,
        'AUC': auc_score_test,

        'Correct Predictions': correct_predictions,
        'Incorrect Predictions': incorrect_predictions
    }

def run_models(df):
    # Data Preparation
    X = df[:, 2:]  # Assuming k-mer counts start from the 3rd column
    y = df['specie']
    
    # # Print class distribution before splitting
    # print("Class distribution before splitting:")
    # print(y.value_counts())

    # Normalize k-mer counts
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (you might want to stratify to preserve class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Print sizes and distributions after splitting
    # print(f"Training set size: {len(y_train)}, Test set size: {len(y_test)}\n")
    # print("Training set class distribution:")
    # print(pl.Series(y_train).value_counts(),"\n")
    # print("Test set class distribution:")
    # print(pl.Series(y_test).value_counts(),"\n")
    local_results = {}
    for model_name, model in models.items():
        log_step(f"Starting evaluation for model: {model_name}.")
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        local_results[model_name] = results
        # print(f"Results for {model_name}:")
        # for metric_name, value in results.items():
        #     if value is not None:
        #         print(f"  {metric_name}: {value:.4f}")
        # print("-" * 40)

    return X_train, X_test, y_train, y_test, local_results

def extract_taxonomy(label):
    """
    Extracts taxonomy from a label string.
    
    If the label contains characters like '[' or ';', or ',', the string is processed accordingly.
    Returns a dictionary with taxonomy levels.
    """
    if '[' in label:
        label = label.split('[')[0]
    if ';' in label:
        parts = label.split(';')
    elif ',' in label:
        label = label.split(',')[0]
        parts = label.split(' ')
        if ':' in label:
            label = label.split(':')[1].strip()
            parts = label.split(' ')
    else:
        parts = label.split(' ')
    taxonomy = {}
    if '__' in label:
        for part in parts:
            level, value = part.split('__') if '__' in part else (part, "")
            taxonomy[level] = value
    else:
        taxonomy['s'] = ' '.join(parts[:2])
    return taxonomy

def extract_species_from_filename(filename):
    """
    Extracts the species name from the filename.
    Assumes the format:
      <prefix>_<genus>_<epithet>_[optional: 'sub' or 'subsp', <sub_epithet>]_[other parts...]
    Examples:
      "1A_Fervidacithiobacillus_caldus_6"  -> "Fervidacithiobacillus caldus"
      "3D_Acidithiobacillus_thiooxidans_sub_albertens"  -> "Acidithiobacillus thiooxidans"
    """
    if '.' in filename:
        filename = filename.split('.')[0]
    parts = filename.split('_')
    if len(parts) < 3:
        return filename
    genus = parts[1]
    species = parts[2]
    return f"{genus} {species}"

def extract_species_from_csv(df, df_species):
    df = df.with_columns(
        pl.col("file")
        .str.extract(r"^([^_]+_[^_]+)", 0)
        .alias("accession")
    )

    df = df.join(df_species, on="accession", how="left")
    df = df.rename({"species": "specie"})

    cols = df.columns
    cols.remove("specie")
    cols.insert(1, "specie")
    df = df.select(cols)
    df = df.drop("label")
    df = df.drop("accession")
    
    rows_before = df.height
    df = df.drop_nulls(subset=["specie"])
    rows_after = df.height
    log_step(f"Rows eliminated: {rows_before - rows_after}")
    log_step(f"Dataset shape: ({rows_after}, {len(df.columns)})")
    
    return df