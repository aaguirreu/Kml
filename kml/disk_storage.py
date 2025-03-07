import os
import json
import tempfile
import numpy as np
import joblib
import polars as pl
from typing import Dict, List, Any

# Directory to store temporary results
TEMP_DIR = os.path.join(tempfile.gettempdir(), "kml_results")

# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def ensure_temp_dir():
    """Ensure the temporary directory exists."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

def save_result(vectorization_name: str, model_results: Dict[str, Dict[str, Any]]):
    """
    Save model results to disk.
    
    Parameters:
        vectorization_name: Name of the vectorization method
        model_results: Dictionary with model results
    """
    ensure_temp_dir()
    
    # Generate a unique filename
    result_id = len(os.listdir(TEMP_DIR))
    filename = os.path.join(TEMP_DIR, f"result_{result_id}.json")
    
    # Save the result to disk using the custom encoder
    with open(filename, 'w') as f:
        json.dump({vectorization_name: model_results}, f, cls=NumpyEncoder)

def load_all_results() -> List[Dict]:
    """
    Load all results from disk.
    
    Returns:
        List of dictionaries with results
    """
    ensure_temp_dir()
    
    results = []
    for filename in sorted(os.listdir(TEMP_DIR)):
        if filename.endswith('.json'):
            filepath = os.path.join(TEMP_DIR, filename)
            with open(filepath, 'r') as f:
                results.append(json.load(f))
    
    return results

def clear_results():
    """Remove all temporary result files."""
    ensure_temp_dir()
    
    for filename in os.listdir(TEMP_DIR):
        if filename.endswith('.json'):
            os.remove(os.path.join(TEMP_DIR, filename))

def save_model(vectorization_name: str, model_name: str, model, output_dir: str):
    """
    Save a trained model to disk with filename format: vectorization_model.joblib
    
    Parameters:
        vectorization_name: Name of the vectorization method
        model_name: Name of the model
        model: Trained model object
        output_dir: Directory where to save the model
    """
    # Create safe filenames by replacing spaces and special characters
    safe_vectorization = vectorization_name.replace(" ", "_")
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    
    # Generate filename with pattern: vectorization_model.joblib
    filename = f"{safe_vectorization}_{safe_model_name}.joblib"
    filepath = os.path.join(output_dir, filename)
    
    # Save the model to disk
    joblib.dump(model, filepath)
    return filepath

def save_prediction_results(vectorization_name: str, model_name: str, 
                          file_names, true_labels, predicted_labels, output_dir: str):
    """
    Save prediction results to a CSV file with format:
    - accession: The original file accession ID
    - true_species: The actual species label
    - predicted_species: The species predicted by the model
    - correct_prediction: Boolean indicating if the prediction was correct
    
    Parameters:
        vectorization_name: Name of the vectorization method
        model_name: Name of the model
        file_names: Array of file names/accessions for the test samples
        true_labels: Array of true species labels
        predicted_labels: Array of predicted species labels
        output_dir: Directory where to save the CSV
    """
    # Create safe filenames by replacing spaces and special characters
    safe_vectorization = vectorization_name.replace(" ", "_")
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    
    # Generate filename with pattern: vectorization_model_predictions.csv
    filename = f"{safe_vectorization}_{safe_model_name}_predictions.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Extract accessions from file names (assuming format like "prefix_accession_...")
    accessions = []
    for file in file_names:
        parts = str(file).split('_')
        if len(parts) >= 2:
            accessions.append(parts[0] + "_" + parts[1])  # Take first two parts as accession
        else:
            accessions.append(file)  # Use whole filename if it doesn't match expected format
    
    # Create DataFrame with results
    df = pl.DataFrame({
        "accession": accessions,
        "true_species": true_labels,
        "predicted_species": predicted_labels,
        "correct_prediction": [t == p for t, p in zip(true_labels, predicted_labels)]
    })
    
    # Save to CSV
    df.write_csv(filepath)
    return filepath
