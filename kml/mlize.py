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
import time  # Add time module import

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

def evaluate_model(model, X_train, X_test, y_train, y_test, auc=True, cv=5):  # Changed default from 5 to 3
    """
    Train and evaluate a model using cross-validation and a test set.

    Parameters:
        model: The machine learning model to evaluate.
        X_train, X_test: Feature sets for training and testing.
        y_train, y_test: Labels for training and testing.
        auc (bool): Whether to compute AUC (only valid if predict_proba is available).
        cv (int): Number of folds for cross-validation.

    Returns:
        tuple: (metrics_dict, prediction_details)
        - metrics_dict: Dictionary containing evaluation metrics
        - prediction_details: Dictionary with 'true_labels' and 'predicted_labels'
    """
    # --- 1) CROSS-VALIDATION ---
    bacc_scorer = make_scorer(balanced_accuracy_score)
    mcc_scorer  = make_scorer(matthews_corrcoef)

    # Track cross-validation time
    cv_start_time = time.time()
    # Average of Balanced Accuracy in CV - use parallel processing with n_jobs=-1
    bacc_cv = cross_val_score(model, X_train, y_train, scoring=bacc_scorer, cv=cv, n_jobs=-1).mean()
    # Average of MCC in CV - use parallel processing with n_jobs=-1
    mcc_cv  = cross_val_score(model, X_train, y_train, scoring=mcc_scorer, cv=cv, n_jobs=-1).mean()
    cv_time = time.time() - cv_start_time

    # --- 2) TRAIN FINAL MODEL on all X_train ---
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # --- 3) EVALUATION on TEST ---
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    start_time = time.time()
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    proba_time = time.time() - start_time
    # Add proba time to prediction time if available
    if y_pred_proba is not None:
        prediction_time += proba_time

    # Test metrics
    accuracy_test = accuracy_score(y_test, y_pred)
    bacc_test     = balanced_accuracy_score(y_test, y_pred)
    f1_test       = f1_score(y_test, y_pred, average='weighted')
    mcc_test      = matthews_corrcoef(y_test, y_pred)

    # AUC in test (if applicable)
    auc_score_test = None
    if auc and y_pred_proba is not None:
        if y_pred_proba.shape[1] == 2:
            # For binary, use the prob. of the positive class
            y_pred_proba = y_pred_proba[:, 1]
        auc_score_test = roc_auc_score(
            y_test,
            y_pred_proba,
            multi_class='ovr' if len(np.unique(y_test)) > 2 else None
        )

    correct_predictions    = np.sum(np.array(y_test) == np.array(y_pred))
    incorrect_predictions  = len(y_test) - correct_predictions

    # --- 4) RETURN RESULTS ---
    metrics = {
        # -- CV Metrics --
        'BACC_CV': bacc_cv,
        'MCC_CV': mcc_cv,
        'CV Time': cv_time,  # Add CV time to results

        # -- Test Metrics --
        'BACC_Test': bacc_test,
        'MCC_Test': mcc_test,

        'Accuracy': accuracy_test,
        'F1 Score': f1_test,
        'AUC': auc_score_test,

        'Correct Predictions': correct_predictions,
        'Incorrect Predictions': incorrect_predictions,
        
        # -- Time Metrics --
        'Training Time': training_time,
        'Prediction Time': prediction_time
    }
    
    # Prepare prediction details for saving to CSV
    prediction_details = {
        'true_labels': y_test,
        'predicted_labels': y_pred
    }
    
    return metrics, prediction_details

def run_models(df, cv_folds=3, output_dir=None, vectorization_name=None, k_value=None):
    # Data Preparation
    X = df[:, 2:]  # Assuming k-mer counts start from the 3rd column
    y = df['specie']
    
    # Store original rows (with 'file' column) for later matching
    original_files = df['file']
    
    # Normalize k-mer counts
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (you might want to stratify to preserve class proportions)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_scaled, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a deep copy of the models dictionary to prevent modifying the original
    models_copy = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
    }
    
    local_results = {}
    
    # Map to store test indices for each model
    test_indices_by_model = {}
    # Map to store prediction details for each model
    prediction_details_by_model = {}
    
    for model_name, model_template in models_copy.items():
        log_step(f"Starting evaluation for model: {model_name}.")
        
        # Make sure we have a valid model instance
        if model_template is None:
            log_step(f"Error: Model {model_name} is not available. Skipping.")
            continue
            
        # Create a fresh instance for each evaluation
        try:
            model = model_template.__class__(**model_template.get_params())
            metrics, prediction_details = evaluate_model(model, X_train, X_test, y_train, y_test, cv=cv_folds)
            
            # Store prediction details for this model
            prediction_details_by_model[model_name] = prediction_details
            # Store test indices for this model
            test_indices_by_model[model_name] = indices_test
            
            # Add k value to results explicitly
            if k_value is not None:
                metrics['K-mer'] = k_value
                
            local_results[model_name] = metrics
            
            # Save model immediately after evaluation if output_dir is provided
            if output_dir and vectorization_name:
                from .disk_storage import save_model, save_prediction_results
                model_path = save_model(vectorization_name, model_name, model, output_dir)
                log_step(f"Model saved to: {model_path}")
                
                # Save prediction results to CSV
                files_test = original_files.to_numpy()[indices_test]
                save_prediction_results(
                    vectorization_name,
                    model_name,
                    files_test,
                    prediction_details['true_labels'],
                    prediction_details['predicted_labels'],
                    output_dir
                )
                log_step(f"Prediction results saved to CSV for {model_name}")
                
                # Force cleanup - Only clear the local reference, not the template
                model = None
                import gc
                gc.collect()
        except Exception as e:
            log_step(f"Error evaluating model {model_name}: {str(e)}")
            continue

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