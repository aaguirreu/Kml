import os
import csv
import subprocess
from .logging import log_memory_usage, log_step, log_results
from .mlize import extract_species_from_filename, extract_species_from_csv, run_models
from .vectorize import vectorization_methods
from . import __date__
from .disk_storage import save_result, clear_results
from .results import plot_grouped_bar, plot_roc_curve_by_model, results_to_df, plot_correct_incorrect_bar, plot_bacc_mcc_4panels, prepare_plot_df, best_k_accuracy
import polars as pl

def evaluate_all_vectorizations(k, df, output_dir):
    """
    Iterates through the vectorization methods defined in 'vectorization_methods'
    and, for each one, runs the ML pipeline by calling 'run_models'.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame with "file", "specie", and k-mer counts.
    
    Returns:
        pl.DataFrame: A table with the results (metrics) for each vectorization method and model.
    """
    for vectorization_name, vectorization_function in vectorization_methods.items():
        # log_memory_usage()
        log_step(f"Applying vectorization: {vectorization_name}")
        vectorized_data = vectorization_function(df)
        log_step(f"Running models for {vectorization_name}")
        X_train, X_test, y_train, y_test, model_results = run_models(vectorized_data)
        model_results = {
            model_name: {'K-mer': k, **metrics}
            for model_name, metrics in model_results.items()
        }
        # Save results to disk instead of storing in memory
        save_result(vectorization_name, model_results)
        plot_df = prepare_plot_df(model_results)
        plot_grouped_bar(plot_df, vectorization_name, output_dir)
        # plot_roc_curve_by_model(X_test, y_test, output_dir)
        plot_correct_incorrect_bar(plot_df, vectorization_name, output_dir)

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

def run_all(args, input_source, k_range):
    """
    Por cada valor de k en k_range, invoca la herramienta externa 'kmertools'
    usando input_source (archivo o directorio). Si la vectorización está habilitada,
    lee el archivo TSV generado y lo envía a evaluate_all_vectorizations.
    """
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
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                log_step(f"Successfully executed kmertools for {k}-mer")
            except subprocess.CalledProcessError as e:
                log_step(f"An error occurred while executing kmertools for {k}-mer analysis. Details: {e.stderr}")
                continue

        if using_vectorization:
            output_filepath = os.path.join(output_dir, 'kmers.counts')
            try:
                log_step(f"Reading {k}-mer file {output_filepath}")
                # Leer el archivo TSV usando Polars (modo eager)
                df = pl.read_csv(output_filepath, separator='\t')
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
                
                log_step(f"Initiating vectorization for {k}-mer")
                evaluate_all_vectorizations(k, df, output_dir)
                log_step(f"Successfully completed processing for {k}-mer analysis\n")
            except Exception as e:
                log_step(f"An error occurred during vectorization for {k}-mer processing: {e}")
                log_results(args.output_dir, input_source, k, k_range)
                return

    # Determinar el mejor k usando la función auxiliar
    if using_vectorization:
        try:
            best_k = best_k_accuracy()
            if best_k is not None:
                log_results(args.output_dir, input_source, best_k, k_range)
            else:
                log_step("Could not determine best k-mer size. Using default.")
                best_k = list(k_range)[-1] if k_range else None
                
            # Generate the metrics file
            metrics_output_filepath = os.path.join(args.output_dir, 'performance_metrics.tsv')
            try:
                # Get results from disk using results_to_df
                results_df = results_to_df()
                if not results_df.is_empty():
                    # Save the DataFrame to a TSV file
                    results_df.write_csv(metrics_output_filepath, separator='\t')
                    log_step(f"Performance metrics have been successfully saved to {metrics_output_filepath}")
                    
                    # Generate the summary plot
                    try:
                        plot_bacc_mcc_4panels(results_df, args.output_dir)
                        log_step("Successfully generated BACC/MCC 4-panel plot")
                    except Exception as e:
                        log_step(f"Error generating BACC/MCC plot: {str(e)}")
                else:
                    log_step("No results available to save to metrics file")
            except Exception as e:
                log_step(f"An error occurred while saving the performance metrics file: {str(e)}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            log_step(f"Error in final result processing: {str(e)}")
            import traceback
            traceback.print_exc()
