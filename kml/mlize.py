from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import polars as pl
import numpy as np
from .logging import log_step
import re
import time  # Add time module import
import gc
import psutil

# Define the models to evaluate.
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),  # Removed use_label_encoder
}

# Function to check memory and adjust model parameters
def get_memory_optimized_model(model_name, n_features, n_samples):
    """
    Creates a memory-optimized version of the specified model based on dataset size.
    
    Parameters:
        model_name: Name of the model
        n_features: Number of features in the dataset
        n_samples: Number of samples in the dataset
        
    Returns:
        A configured sklearn model with memory-optimized parameters
    """
    # Get current memory state
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    log_step(f"Available memory: {available_gb:.2f} GB")
    
    # Estimate dataset size in memory (rough calculation)
    dataset_size_gb = (n_samples * n_features * 8) / (1024 ** 3)  # 8 bytes per float64
    log_step(f"Estimated dataset size: {dataset_size_gb:.2f} GB")
    
    # For extremely large feature sets (>20k features)
    high_dimensional = n_features > 20000
    very_high_dimensional = n_features > 30000
    
    if model_name == 'Random Forest':
        if very_high_dimensional:
            # For extremely high dimensional data, use extreme memory optimization
            log_step("Using extremely memory-optimized Random Forest (max_depth=10, n_estimators=50)")
            return RandomForestClassifier(
                n_estimators=50,  # Reduce number of trees
                max_depth=10,     # Limit tree depth
                min_samples_split=10,  # Require more samples to split
                min_samples_leaf=4,   # Require more samples per leaf
                max_features='sqrt',  # Use sqrt of features (reduces memory)
                bootstrap=True,      # Use bootstrapping (reduces memory)
                n_jobs=1,            # Use only one core (slower but less memory)
                random_state=42,
                verbose=1            # Show progress
            )
        elif high_dimensional:
            # For high dimensional data, use moderate memory optimization
            log_step("Using memory-optimized Random Forest (max_depth=20, n_estimators=80)")
            return RandomForestClassifier(
                n_estimators=80,  # Slightly reduce number of trees
                max_depth=20,     # Limit tree depth
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',  # Use sqrt of features
                bootstrap=True,
                n_jobs=2,         # Use limited parallelism
                random_state=42,
                verbose=1         # Show progress
            )
        else:
            # For regular datasets, use slight optimization
            return RandomForestClassifier(
                n_estimators=100,
                max_features='sqrt',  # Still use feature selection
                n_jobs=-1,           # Use all cores
                random_state=42
            )
    
    elif model_name == 'Logistic Regression':
        if high_dimensional:
            log_step("Using memory-optimized Logistic Regression (saga solver)")
            return LogisticRegression(
                solver='saga',        # Memory efficient solver
                penalty='l1',         # L1 regularization for feature selection
                C=0.1,               # Stronger regularization 
                max_iter=500,        # Fewer iterations for memory conservation
                tol=1e-3,            # Looser tolerance
                random_state=42
            )
        else:
            return LogisticRegression(max_iter=1000, random_state=42)
    
    elif model_name == 'Support Vector Machine (RBF Kernel)':
        if high_dimensional:
            log_step("Using memory-optimized SVM (linear kernel)")
            return SVC(
                kernel='linear',     # Linear kernel is more memory efficient
                C=1.0,
                probability=True,
                random_state=42,
                cache_size=1000      # Limit cache size
            )
        else:
            return SVC(kernel='rbf', probability=True, random_state=42)
    
    # Default case - return the original model
    return models.get(model_name)

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
    # Add memory tracking
    initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    log_step(f"Memory usage before model evaluation: {initial_memory:.2f} MB")
    
    # --- 1) CROSS-VALIDATION ---
    bacc_scorer = make_scorer(balanced_accuracy_score)
    mcc_scorer  = make_scorer(matthews_corrcoef)

    # Track cross-validation time
    cv_start_time = time.time()
    
    try:
        # Average and std of Balanced Accuracy in CV
        n_jobs = 1 if X_train.shape[1] > 20000 else -1
        log_step(f"Running cross-validation with {cv} folds and n_jobs={n_jobs}")
        bacc_cv_scores = cross_val_score(model, X_train, y_train, scoring=bacc_scorer, cv=cv, n_jobs=n_jobs)
        bacc_cv = bacc_cv_scores.mean()
        bacc_cv_std = bacc_cv_scores.std()
        # Average and std of MCC in CV
        mcc_cv_scores = cross_val_score(model, X_train, y_train, scoring=mcc_scorer, cv=cv, n_jobs=n_jobs)
        mcc_cv = mcc_cv_scores.mean()
        mcc_cv_std = mcc_cv_scores.std()
    except MemoryError:
        log_step("Memory error during cross-validation. Using reduced CV.")
        bacc_cv_scores = cross_val_score(model, X_train, y_train, scoring=bacc_scorer, cv=3, n_jobs=1)
        bacc_cv = bacc_cv_scores.mean()
        bacc_cv_std = bacc_cv_scores.std()
        mcc_cv_scores = cross_val_score(model, X_train, y_train, scoring=mcc_scorer, cv=3, n_jobs=1)
        mcc_cv = mcc_cv_scores.mean()
        mcc_cv_std = mcc_cv_scores.std()
    except Exception as e:
        log_step(f"Error during cross-validation: {str(e)}")
        bacc_cv = 0
        bacc_cv_std = 0
        mcc_cv = 0
        mcc_cv_std = 0
        
    cv_time = time.time() - cv_start_time
    log_step(f"Cross-validation completed in {cv_time:.2f} seconds")
    
    # Force garbage collection after CV
    gc.collect()

    # --- 2) TRAIN FINAL MODEL on all X_train ---
    start_time = time.time()
    try:
        model.fit(X_train, y_train)
    except MemoryError:
        log_step("Memory error during training. Attempting with sample reduction.")
        # Try with a smaller subset if memory error occurs
        sample_size = min(10000, X_train.shape[0])
        indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
        X_train_sample = X_train[indices]
        y_train_sample = np.array(y_train)[indices]
        model.fit(X_train_sample, y_train_sample)
        
    training_time = time.time() - start_time
    log_step(f"Model training completed in {training_time:.2f} seconds")
    
    # Force garbage collection after training
    gc.collect()

    # --- 3) EVALUATION on TEST ---
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Only calculate predict_proba if the model supports it and won't cause memory issues
    y_pred_proba = None
    proba_time = 0
    if hasattr(model, "predict_proba"):
        try:
            start_time = time.time()
            y_pred_proba = model.predict_proba(X_test)
            proba_time = time.time() - start_time
            # Add proba time to prediction time if available
            prediction_time += proba_time
        except MemoryError:
            log_step("Memory error during predict_proba. Skipping probability calculation.")
        except Exception as e:
            log_step(f"Error during predict_proba: {str(e)}")
    
    log_step(f"Prediction completed in {prediction_time:.2f} seconds")

    # Test metrics
    accuracy_test = accuracy_score(y_test, y_pred)
    bacc_test     = balanced_accuracy_score(y_test, y_pred)
    f1_test       = f1_score(y_test, y_pred, average='weighted')
    mcc_test      = matthews_corrcoef(y_test, y_pred)

    # AUC in test (if applicable)
    auc_score_test = None
    if auc and y_pred_proba is not None:
        try:
            if y_pred_proba.shape[1] == 2:
                # For binary, use the prob. of the positive class
                y_pred_proba = y_pred_proba[:, 1]
            auc_score_test = roc_auc_score(
                y_test,
                y_pred_proba,
                multi_class='ovr' if len(np.unique(y_test)) > 2 else None
            )
        except Exception as e:
            log_step(f"Error calculating AUC: {str(e)}")

    correct_predictions    = np.sum(np.array(y_test) == np.array(y_pred))
    incorrect_predictions  = len(y_test) - correct_predictions
    
    # Track final memory usage
    final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    memory_increase = final_memory - initial_memory
    log_step(f"Memory usage after model evaluation: {final_memory:.2f} MB (increase: {memory_increase:.2f} MB)")

    # --- 4) RETURN RESULTS ---
    metrics = {
        # -- CV Metrics --
        'BACC_CV': bacc_cv,
        'BACC_CV_STD': bacc_cv_std,
        'MCC_CV': mcc_cv,
        'MCC_CV_STD': mcc_cv_std,
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
    
    # Get dataset dimensions for memory optimization
    n_samples, n_features = X.shape
    log_step(f"Dataset for modeling: {n_samples} samples, {n_features} features")
    
    # Normalize k-mer counts
    log_step("Normalizing features with StandardScaler")
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except MemoryError:
        log_step("Memory error during scaling. Using batch processing.")
        # Process in batches for large datasets
        batch_size = 5000
        X_list = []
        
        # Fit scaler on first batch
        first_batch = X[:min(batch_size, n_samples)]
        scaler.fit(first_batch)
        
        # Transform in batches
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X[i:end]
            X_batch_scaled = scaler.transform(X_batch)
            X_list.append(X_batch_scaled)
            del X_batch  # Free memory
            gc.collect()
            
        X_scaled = np.vstack(X_list)
        del X_list  # Free memory
        gc.collect()

    # Train-test split (you might want to stratify to preserve class proportions)
    log_step("Performing train-test split")
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_scaled, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    
    # Free memory after splitting
    del X_scaled
    gc.collect()
    
    # Create a deep copy of the models dictionary to prevent modifying the original
    models_copy = {
        'Random Forest': None,  # Will be initialized with optimized parameters
        'Logistic Regression': None,
        'Support Vector Machine (RBF Kernel)': None,
        'XGBoost': None,
    }
    
    local_results = {}
    
    # Map to store test indices for each model
    test_indices_by_model = {}
    # Map to store prediction details for each model
    prediction_details_by_model = {}
    
    for model_name in models_copy.keys():
        log_step(f"Starting evaluation for model: {model_name}.")
        
        # Get memory-optimized model for this dataset
        model = get_memory_optimized_model(model_name, n_features, n_samples)
        
        if model is None:
            log_step(f"Error: Model {model_name} is not available or could not be optimized. Skipping.")
            continue

        try:
            # Para XGBoost: necesitamos codificar las etiquetas numéricamente
            if model_name == 'XGBoost':
                log_step("XGBoost requires numerical labels. Encoding labels.")
                # Crear un LabelEncoder para XGBoost
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train)
                y_test_encoded = label_encoder.transform(y_test)
                
                # Crear un mapeo para recuperar las etiquetas originales después
                class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
                log_step(f"Encoded {len(class_mapping)} unique classes for XGBoost")
                
                # Evaluar XGBoost con etiquetas codificadas
                log_step(f"Training and evaluating {model_name}")
                metrics, prediction_details = evaluate_model(model, X_train, X_test, y_train_encoded, y_test_encoded, cv=cv_folds)
                
                # Convertir las predicciones numéricas a las etiquetas originales
                prediction_details['true_labels'] = np.array([class_mapping[label] for label in prediction_details['true_labels']])
                prediction_details['predicted_labels'] = np.array([class_mapping[label] for label in prediction_details['predicted_labels']])
            else:
                # Para los demás modelos, usar etiquetas originales
                log_step(f"Training and evaluating {model_name}")
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
                
                # Generate ROC curve if model supports predict_proba
                if hasattr(model, "predict_proba") and output_dir:
                    from .results import plot_roc_curve_by_model
                    try:
                        # Generate ROC curve using the already fitted model
                        log_step(f"Generating ROC curve for {model_name}")
                        plot_roc_curve_by_model(
                            X_test, 
                            prediction_details['true_labels'], 
                            model,  # Pass the fitted model
                            model_name, 
                            vectorization_name,
                            output_dir
                        )
                    except Exception as e:
                        log_step(f"Error generating ROC curve for {model_name}: {str(e)}")
                
                # Force cleanup - Only clear the local reference, not the template
                model = None
                gc.collect()
            
            # Log memory usage after each model
            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            log_step(f"Memory usage after {model_name}: {memory_usage:.2f} MB")
            
        except Exception as e:
            log_step(f"Error evaluating model {model_name}: {str(e)}")
            # Print full traceback for debugging
            import traceback
            log_step(traceback.format_exc())
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