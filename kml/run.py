import os
import csv
import subprocess
import time
from .logging import log_memory_usage, log_step, log_results
from .mlize import extract_species_from_filename, extract_species_from_csv, run_models
from .vectorize import vectorization_methods, apply_pca
from . import __date__
from .disk_storage import save_result, clear_results, save_model, get_predictions_csv_path, load_model, get_model_path, ensure_global_plots_dir
from .results import plot_grouped_bar, plot_roc_curve_by_model, results_to_df, plot_correct_incorrect_bar, plot_bacc_mcc_4panels, prepare_plot_df, best_k_accuracy, plot_confusion_matrices, plot_feature_importance_heatmap, plot_prediction_pie
import polars as pl

def evaluate_all_vectorizations(k, df, output_dir, apply_pca_flag=False, pca_components=50):
    """
    Iterates through the vectorization methods defined in 'vectorization_methods'
    and, for each one, runs the ML pipeline by calling 'run_models'.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame with "file", "specie", and k-mer counts.
        output_dir (str): Directory to save outputs
        apply_pca_flag (bool): Whether to apply PCA to each vectorization method
        pca_components (int): Number of PCA components if apply_pca_flag is True
    
    Returns:
        pl.DataFrame: A table with the results (metrics) for each vectorization method and model.
    """
    # Track total times for final summary
    total_vectorization_time = 0
    total_training_time = 0
    total_prediction_time = 0
    total_models_count = 0
    total_cv_time = 0
    
    for vectorization_name, vectorization_function in vectorization_methods.items():
        # log_memory_usage()
        log_step(f"Applying vectorization: {vectorization_name}")
        
        # Time the vectorization process
        vectorization_start = time.time()
        
        # Apply vectorization - the time is now tracked by the decorator
        vectorized_data = vectorization_function(df)
        
        # Extract vectorization time from the DataFrame (added by the decorator)
        vectorization_time = getattr(vectorized_data, '_vectorization_time', 0)
        total_vectorization_time += vectorization_time
        log_step(f"Vectorization time for {vectorization_name}: {vectorization_time:.4f} seconds")
        
        # Apply PCA if the flag is set
        if apply_pca_flag:
            log_step(f"Applying PCA to {vectorization_name}")
            original_vectorization_name = vectorization_name
            
            # Calculate number of features for logging purposes
            num_features = len(vectorized_data.columns) - 2  # Subtract 'file' and 'specie' columns
            
            # Pass None for n_components to use the default 90% if pca_components is -1 (auto)
            # Otherwise use the user-specified value
            use_components = None if pca_components == -1 else pca_components
            
            # Apply PCA with dynamic or user-specified components
            vectorized_data = apply_pca(vectorized_data, n_components=use_components)
            
            # Calculate actual number of components used
            actual_components = len(vectorized_data.columns) - 2  # Subtract 'file' and 'specie' columns
            
            # Update the vectorization name to include PCA
            vectorization_name = f"{original_vectorization_name} + PCA({actual_components})"
            
            # Log PCA-specific information
            pca_time = getattr(vectorized_data, '_pca_time', 0)
            total_vectorization_time += pca_time
            log_step(f"PCA time for {vectorization_name}: {pca_time:.4f} seconds")
            log_step(f"Using {actual_components} components (90% of {num_features} features)")
            
            variance_explained = getattr(vectorized_data, '_variance_explained', 0)
            log_step(f"PCA variance explained: {variance_explained:.2%}")
        
        log_step(f"Running models for {vectorization_name}")
        # Pass output_dir and vectorization_name to run_models so it can save models
        X_train, X_test, y_train, y_test, model_results = run_models(
            vectorized_data, 
            output_dir=output_dir, 
            vectorization_name=vectorization_name,
            k_value=k  # Explicitly pass k value to run_models
        )
        
        # Track training and prediction times for summary
        for model_name, metrics in model_results.items():
            # Calculate model-specific time including its portion of vectorization time
            # This ensures each model has its own complete processing time
            model_time = (
                vectorization_time +  # Time for vectorization that applies to all models
                metrics.get('CV Time', 0) +  # Model-specific CV time
                metrics.get('Training Time', 0) +  # Model-specific training time
                metrics.get('Prediction Time', 0)  # Model-specific prediction time
            )
            
            # Update existing time metrics
            total_cv_time += metrics.get('CV Time', 0)
            total_training_time += metrics.get('Training Time', 0)
            total_prediction_time += metrics.get('Prediction Time', 0)
            total_models_count += 1
            
            # Log the complete model processing time
            log_step(f"Model: {model_name} - Complete Processing Time: {model_time:.4f}s")
            log_step(f"  ├─ Vectorization Time: {vectorization_time:.4f}s")
            log_step(f"  ├─ Cross-Validation Time: {metrics.get('CV Time', 0):.4f}s")
            log_step(f"  ├─ Training Time: {metrics.get('Training Time', 0):.4f}s")
            log_step(f"  └─ Prediction Time: {metrics.get('Prediction Time', 0):.4f}s")
            
            # Add the complete model time to metrics
            metrics['Complete Model Time'] = model_time
        
        # Add k-mer value to results explicitly - ensure each model has the k value
        for model_name in model_results:
            model_results[model_name]['K-mer'] = k
        
        # Save results and create plots
        save_result(vectorization_name, model_results)
        plot_df = prepare_plot_df(model_results)
        plot_grouped_bar(plot_df, vectorization_name, output_dir)
        plot_correct_incorrect_bar(plot_df, vectorization_name, output_dir)
        
        # Generate additional visualization for each model
        for model_name in model_results.keys():
            try:
                # Load saved predictions using the updated path function
                predictions_path = get_predictions_csv_path(vectorization_name, model_name, output_dir)
                if os.path.exists(predictions_path):
                    predictions_df = pl.read_csv(predictions_path)
                    y_true = predictions_df["true_species"].to_numpy()
                    y_pred = predictions_df["predicted_species"].to_numpy()
                    
                    # Generate confusion matrices
                    log_step(f"Generating confusion matrix for {vectorization_name} - {model_name}")
                    plot_confusion_matrices(y_true, y_pred, vectorization_name, model_name, output_dir)
                    
                    # Generate pie chart for predictions
                    log_step(f"Generating prediction pie chart for {vectorization_name} - {model_name}")
                    plot_prediction_pie(y_true, y_pred, vectorization_name, model_name, output_dir)
                    
                    # Try to load the saved model for feature importance using the updated path function
                    model_path = get_model_path(vectorization_name, model_name, output_dir)
                    
                    if os.path.exists(model_path):
                        model = load_model(model_path)
                        
                        # Get feature names (all columns except file and specie)
                        feature_names = [col for col in df.columns if col not in ["file", "specie"]]
                        
                        # Generate feature importance heatmap
                        log_step(f"Generating feature importance for {vectorization_name} - {model_name}")
                        success = plot_feature_importance_heatmap(model, feature_names, vectorization_name, model_name, output_dir)
                        if not success:
                            log_step(f"Skipping feature importance visualization for {model_name} - not supported for this model type")
            except Exception as e:
                log_step(f"Error generating visualizations for {model_name}: {str(e)}")
                continue

def filter_single_sample_classes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filtra las especies que tengan menos de 3 muestras en un DataFrame de Polars.
    
    Parameters:
        df (pl.DataFrame): DataFrame de entrada que contiene la columna 'specie'.
    
    Returns:
        pl.DataFrame: DataFrame con solo las especies que tienen al menos 3 muestras.
    """
    # Agrupar por 'specie' y contar el número de ocurrencias
    counts = df.group_by("specie").agg(pl.count().alias("species_count"))
    # Unir la cuenta con el DataFrame original
    df = df.join(counts, on="specie")
    # Filtrar aquellas filas con species_count >= 3
    df = df.filter(pl.col("species_count") >= 3)
    # Eliminar la columna auxiliar
    df = df.drop("species_count")
    return df

def analyze_taxonomy(df):
    """
    Analyze and log the unique species and genera in the dataset.
    
    Parameters:
        df (pl.DataFrame): DataFrame containing a 'specie' column with species names
    """
    if 'specie' not in df.columns:
        log_step("Cannot analyze taxonomy: 'specie' column not found in DataFrame")
        return

    try:
        # Use a safer way to get unique species without assuming a particular data type
        unique_species_list = df['specie'].unique().to_list()
        log_step(f"Number of unique species in the dataset: {len(unique_species_list)}")
        
        # Extract genera using Python native methods rather than Polars operations
        genera_set = set()
        
        for species in unique_species_list:
            try:
                # Handle different potential data types
                if isinstance(species, list):
                    # If the species is a list, take first element as the species name
                    species_str = str(species[0]) if species else ""
                else:
                    # Otherwise convert to string
                    species_str = str(species)
                
                # Extract the genus (first word)
                genus = species_str.split(' ')[0] if species_str else ""
                if genus:
                    genera_set.add(genus)
            except Exception as e:
                log_step(f"Warning: Could not extract genus from '{species}': {str(e)}")
        
        log_step(f"Number of unique genera in the dataset: {len(genera_set)}")
            
    except Exception as e:
        log_step(f"Error in taxonomy analysis: {str(e)}")
        import traceback
        log_step(traceback.format_exc())
        log_step("Continuing without taxonomy analysis")

def save_performance_metrics(output_dir, kmertools_times=None):
    """
    Save performance metrics to a TSV file, handling any exceptions that occur.
    
    Parameters:
        output_dir (str): Directory where to save the metrics file
        kmertools_times (dict, optional): Dictionary mapping k values to kmertools execution times
    
    Returns:
        pl.DataFrame: The DataFrame that was saved (or attempted to be saved)
    """
    metrics_output_filepath = os.path.join(output_dir, 'performance_metrics.tsv')
    results_df = None
    
    try:
        # Get results from disk using results_to_df
        results_df = results_to_df()
        if not results_df.is_empty():
            # Verify K-mer column exists in results
            if "K-mer" not in results_df.columns:
                log_step("Warning: K-mer column not found in results. This indicates a problem with result tracking.")
                # Instead of adding a fixed value, log the missing information
                log_step("Please check that k values are being correctly passed to models.")
                
                # If we still need the file to be created, add the k values but with a clear warning
                results_df = results_df.with_columns(
                    pl.lit(None).cast(pl.Int64).alias("K-mer")
                )
                log_step("Added placeholder K-mer column with null values.")
            
            # Add kmertools counting time if available
            if kmertools_times is not None:
                # Create a function to look up k-mer times
                def get_kmer_time(k):
                    return kmertools_times.get(k, 0.0)
                
                # Add the Counting Time column based on K-mer value
                results_df = results_df.with_columns(
                    pl.col("K-mer").map_elements(get_kmer_time).alias("Counting Time")
                )
                log_step("Added K-mer counting times to performance metrics")
            
            # Reorder columns according to desired format
            time_cols = ["Counting Time", "CV Time", "Training Time", "Prediction Time", "Complete Model Time"]
            main_cols = ["K-mer", "Vectorization", "Model", "Accuracy", "F1 Score", "AUC"]
            other_cols = [col for col in results_df.columns if col not in main_cols + time_cols]
            
            # Ensure we only select columns that exist (Counting Time might not be there)
            time_cols = [col for col in time_cols if col in results_df.columns]
            results_df = results_df.select(main_cols + other_cols + time_cols)
            
            # Write the DataFrame to TSV
            try:
                results_df.write_csv(metrics_output_filepath, separator='\t')
                log_step(f"Performance metrics have been successfully saved to {metrics_output_filepath}")
            except Exception as save_error:
                log_step(f"Error saving performance metrics file: {str(save_error)}")
        else:
            log_step("No results available to save to metrics file")
    except Exception as e:
        log_step(f"An error occurred while preparing the performance metrics: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results_df

def run_all(args, input_source, k_range):
    """
    Por cada valor de k en k_range, invoca la herramienta externa 'kmertools'
    usando input_source (archivo o directorio). Si la vectorización está habilitada,
    lee el archivo TSV generado y lo envía a evaluate_all_vectorizations.
    """

    # Track specific operation times
    model_time = 0
    kmertools_time = 0
    
    # Dictionary to store kmertools time for each k
    kmertools_times = {}
    
    # Clear previous results at the beginning
    clear_results()
    
    # Determinar si se usará vectorización (si se especificó algún método o -va)
    using_vectorization = args.vectorization_all or (args.vectorization and len(args.vectorization.split(',')) > 0)
    
    for k in k_range:
        output_dir = f"{args.output_dir}/{str(k)}-mer/"
        if args.ctr:
            try:
                cmd = [
                    "kmertools", "kml",
                    "-i", input_source,
                    "-o", output_dir,
                    "-k", str(k),
                    "--acgt"
                ]
                log_step(f"Initiating kmertools kml counting for {k}-mer")
                kmertools_start = time.time()  # Start timing kmertools
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                kmertools_end = time.time()    # End timing kmertools
                kmer_execution_time = kmertools_end - kmertools_start
                kmertools_time += kmer_execution_time  # Accumulate kmertools time
                
                # Store the time for this specific k-value
                kmertools_times[k] = kmer_execution_time
                
                log_step(f"Successfully executed kmertools for {k}-mer in {kmer_execution_time:.4f} seconds")
            except subprocess.CalledProcessError as e:
                log_step(f"An error occurred while executing kmertools for {k}-mer analysis. Details: {e.stderr}")
                continue

        if using_vectorization:
            output_filepath = os.path.join(output_dir, 'kmers.counts')
            try:
                log_step(f"Reading {k}-mer file {output_filepath}")
                # Time file reading
                df = pl.read_csv(output_filepath, separator='\t')
                # Force evaluation of the DataFrame to get accurate timing
                df_shape = df.shape
                log_step(f"DataFrame shape: {df_shape}")
            except Exception as e:
                log_step(f"Failed to read the file '{output_filepath}'. Error details: {e}")
                continue

            try:
                # Extraer taxonomía a partir de la columna "file"
                if args.specie_csv:
                    log_step(f"Extracting species from CSV file {args.specie_csv}")
                    df_species = pl.read_csv(args.specie_csv, separator=',', columns=['accession', 'species'])
                    df = extract_species_from_csv(df, df_species)
                else:
                    # NO HA SIDO TESTEADO
                    log_step("Extracting species from filenames")
                    df = df.with_columns(
                        (
                            pl.col("file").str.split("_").arr.get(1) + " " +
                            pl.col("file").str.split("_").arr.get(2)
                        ).alias("label")
                    )
                    log_step("Renaming 'label' column to 'specie'")
                    df = df.rename({"label": "specie"})
                
                log_step("Filtering classes with less than 3 samples")
                df = filter_single_sample_classes(df)
                
                # Analyze and log taxonomy information
                analyze_taxonomy(df)
                
                # Track vectorization and model times separately
                log_step(f"Initiating vectorization for {k}-mer")
                process_start = time.time()
                # Pass PCA flag and components to evaluate_all_vectorizations
                evaluate_all_vectorizations(k, df, output_dir, apply_pca_flag=args.pca, pca_components=args.pca_components)
                process_end = time.time()
                # Track actual model processing time separately from other operations
                model_time += process_end - process_start
                log_step(f"Successfully completed processing for {k}-mer analysis")
                log_step(f"Total processing time for {k}-mer: {process_end - process_start:.4f} seconds\n")
                
            except Exception as e:
                log_step(f"An error occurred during vectorization for {k}-mer processing: {e}")
                # Save performance metrics before exiting due to error
                save_performance_metrics(args.output_dir)
                log_results(args.output_dir, input_source, k, k_range)
                return

    # Determinar el mejor k usando la función auxiliar
    if using_vectorization:
        try:
            # Generate the metrics file first using our new function - now pass kmertools_times
            results_df = save_performance_metrics(args.output_dir, kmertools_times)
            
            # If we got results, continue with post-processing
            if results_df is not None and not results_df.is_empty():
                try:
                    # Now determine best k
                    best_k = best_k_accuracy()
                    if best_k is not None:
                        log_results(args.output_dir, input_source, best_k, k_range)
                    else:
                        log_step("Could not determine best k-mer size. Using default.")
                        best_k = list(k_range)[-1] if k_range else None
                        log_results(args.output_dir, input_source, best_k, k_range)
                    
                    # Create global_plots directory for final results
                    global_plots_dir = ensure_global_plots_dir(args.output_dir)
                    # Call plot_bacc_mcc_4panels with the global plots directory
                    plot_bacc_mcc_4panels(results_df, global_plots_dir)
                    log_step("Successfully generated BACC/MCC 4-panel plot in global_plots directory")
                except Exception as e:
                    log_step(f"Error in post-processing or plot generation: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            log_step(f"Error in final result processing: {str(e)}")
            import traceback
            traceback.print_exc()

