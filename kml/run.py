import os
import csv
import subprocess
import pandas as pd
from .logging import log_memory_usage, log_step, log_results
from .mlize import extract_species_from_filename, run_models
from .vectorize import vectorization_methods
from . import __date__, all_results
from .results import plot_grouped_bar, plot_roc_curve_by_model, results_to_df, plot_correct_incorrect_bar, prepare_plot_df, best_k_accuracy

def evaluate_all_vectorizations(k, df, output_dir):
    """
    Iterates through the vectorization methods defined in 'vectorization_methods'
    and, for each one, runs the ML pipeline by calling 'run_models'.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with "file", "specie", and k-mer counts.
    
    Returns:
        pd.DataFrame: A table with the results (metrics) for each vectorization method and model.
    """
    local_results = []
    for vectorization_name, vectorization_function in vectorization_methods.items():
        log_step(f"Applying vectorization: {vectorization_name}")
        # log_memory_usage()
        vectorized_data = vectorization_function(df)
        X_train, X_test, y_train, y_test, model_results = run_models(vectorized_data)
        model_results = {
            model_name: {'K-mer': k, **metrics}
            for model_name, metrics in model_results.items()
        }
        all_results.append({vectorization_name: model_results})
        plot_df = prepare_plot_df(model_results)
        plot_grouped_bar(plot_df, vectorization_name, output_dir)
        # plot_roc_curve_by_model(X_test, y_test, output_dir)
        plot_correct_incorrect_bar(plot_df, vectorization_name, output_dir)

def filter_single_sample_classes(df):
    """
    Filters out classes (species) that have fewer than 3 samples in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'specie' column.
    
    Returns:
        pd.DataFrame: DataFrame with classes having fewer than 3 samples removed.
    """
    counts = df['specie'].value_counts()
    single_sample_classes = counts[counts < 3].index.tolist()
    if len(single_sample_classes) > 0:
        print(f"Found {len(single_sample_classes)} species with fewer than 3 samples. Removing these species: {single_sample_classes}")
    df = df[df['specie'].isin(counts[counts >= 3].index)]
    return df

def run_all(args, input_source, k_range):
    """
    For each value of k in k_range, invokes the external 'kmertools' tool
    using input_source (file or directory). If vectorization mode is enabled,
    it reads the generated TSV file and sends it to evaluate_all_vectorizations.
    """
    # Determinar si se va a usar vectorización (si se especificó algún método o -va)
    using_vectorization = args.vectorization_all or (args.vectorization and len(args.vectorization.split(',')) > 0)
    
    for k in k_range:
        log_step(f"Initiating kmertools kml counting for {k}-mer")
        output_dir = f"{args.output_dir}/{str(k)}-mer/"
        cmd = [
            "kmertools", "kml",
            "-i", input_source,
            "-o", output_dir,
            "-k", str(k),
            "--acgt"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            log_step(f"Successfully executed kmertools for {k}-mer")
        except subprocess.CalledProcessError as e:
            log_step(f"An error occurred while executing kmertools for {k}-mer analysis. Details: {e.stderr}")
            continue

        output_filepath = os.path.join(output_dir, 'kmers.counts')
        try:
            df = pd.read_csv(output_filepath, sep='\t')
            # log_step(f"File {output_filepath} read successfully.")
        except Exception as e:
            log_step(f"Failed to read the file '{output_filepath}'. Error details: {e}")
            continue

        if using_vectorization:
            try:
                # Extract taxonomy from the 'label' column
                df["label"] = df["file"].apply(extract_species_from_filename)
                df = df.rename(columns={'label': 'specie'})
                df = filter_single_sample_classes(df)
                evaluate_all_vectorizations(k, df, output_dir)
            except Exception as e:
                log_step(f"An error occurred during vectorization for {k}-mer processing: {e}")
                log_results(args.output_dir, input_source, k, k_range)
                return
            log_step(f"Successfully completed processing for {k}-mer analysis\n")

    # Determine the best k using our helper function
    if using_vectorization:
        best_k = best_k_accuracy()
        log_results(args.output_dir, input_source, best_k, k_range)

        metrics_output_filepath = os.path.join(args.output_dir, f'performance_metrics.tsv')
        try:
            results_to_df().to_csv(metrics_output_filepath, sep='\t', index=False)
            log_step(f"Performance metrics have been successfully saved to {metrics_output_filepath}")
        except Exception as e:
            log_step(f"An error occurred while saving the performance metrics file: {e}")
