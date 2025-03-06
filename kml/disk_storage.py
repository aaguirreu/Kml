import os
import json
import tempfile
import numpy as np
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
