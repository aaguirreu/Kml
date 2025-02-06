#!/usr/bin/env python3
import time
import os
import argparse
import traceback
import sys
import pandas as pd
from . import __date__
from .logging import log_step
from .run import run_all, evaluate_all_vectorizations
from .mlize import models  # Import the models to allow filtering based on user input
from .vectorize import vectorization_methods

def parse_k_range(k_str):
    """
    Takes a string representing either a single value (e.g., "3")
    or a range (e.g., "2-12") and returns a range object.
    """
    if '-' in k_str:
        k_start, k_end = map(int, k_str.split('-'))
        if k_start >= k_end:
            raise ValueError("The first number in the range must be less than the second")
        return range(k_start, k_end + 1)
    else:
        k = int(k_str)
        return range(k, k + 1)

def check_tsv(file_path):
    """
    Validates that the file is a valid .tsv and contains at least the first two columns:
    'file' and 'label', as well as additional columns for k-mers.
    """
    if not file_path.endswith('.tsv'):
        log_step(f"Error: The file {file_path} is not a .tsv.")
        log_step("Exiting the program")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, sep='\t')
        required_columns = ['file', 'label']
        if not all(col in df.columns[:2] for col in required_columns):
            log_step(f"Error: The file {file_path} must contain the columns {required_columns} in this exact order at the beginning.")
            log_step("Exiting the program")
            sys.exit(1)

        if len(df.columns) <= 2:
            log_step(f"Error: The file {file_path} must contain additional columns corresponding to k-mers.")
            log_step("Exiting the program")
            sys.exit(1)

        log_step(f"File {file_path} successfully validated.")
        return df
    except Exception as e:
        log_step(f"Error reading the file {file_path}: {e}")
        log_step("Exiting the program")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="""Process sequence files with k-mer vectorization

Available options:
  Vectorization methods (-v):
    KMER    - K-mer Frequency
    TFIDF   - TF-IDF
    RF      - Relative Frequency
    HASH    - K-mer Hashing

  Machine Learning models (-m):
    RF      - Random Forest
    LR      - Logistic Regression
    SVM     - Support Vector Machine (RBF Kernel)

Examples:
  # Use specific vectorization methods and models:
    python -m kmer -k 3 -v KMER,TFIDF -m RF,LR -f input.fasta -o output/
  
  # Use all available methods and models:
    python -m kmer -k 3 -va -ma -f input.fasta -o output/
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-k', help='Value of k for k-mers (single value or range, e.g., 3 or 2-12)', required=True)
    parser.add_argument('-v', '--vectorization', type=str,
                       help='Comma-separated list of vectorization methods: kmer,tf-idf,RF,hash,oh')
    parser.add_argument('-f', '--file', type=str,
                        help='Path to a sequence file (.fasta, .fna, .fa, .fastq or .tsv in vectorization)')
    parser.add_argument('-d', '--directory', type=str,
                        help='Path to a directory containing sequence files')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory for output files')
    # Nueva flag para escoger los modelos deseados: ejemplo: RF,LR,SVM
    parser.add_argument('-m', '--models', type=str, help='Comma-separated list of models to evaluate: RF,LR,SVM')
    parser.add_argument('-va', '--vectorization-all', action='store_true',
                       help='Use all available vectorization methods')
    parser.add_argument('-ma', '--models-all', action='store_true',
                       help='Use all available models')
    args = parser.parse_args()

    # Si se especifica la flag --models o --models-all, filtrar el diccionario de modelos
    if args.models or args.models_all:
        if args.models_all:
            log_step("Using all available models.")
            # No hacemos nada, dejamos models como está
        else:
            code_to_name = {
                "RF": "Random Forest",
                "LR": "Logistic Regression",
                "SVM": "Support Vector Machine (RBF Kernel)"
            }
            selected_codes = [code.strip().upper() for code in args.models.split(',')]
            new_models = {}
            for code in selected_codes:
                key = code_to_name.get(code)
                if key and key in models:
                    new_models[key] = models[key]
                else:
                    log_step(f"Warning: Unknown model code '{code}'")
            
            if new_models:  # Solo actualizar si se encontraron modelos válidos
                models.clear()
                models.update(new_models)
            else:
                log_step("No valid models selected. Using all available models.")

    # Procesar los métodos de vectorización seleccionados
    if args.vectorization or args.vectorization_all:
        if args.vectorization_all:
            log_step("Using all available vectorization methods.")
            # No hacemos nada, dejamos vectorization_methods como está
        else:
            code_to_name = {
                "KMER": "K-mer Frequency",
                "TFIDF": "TF-IDF",
                "RF": "Relative Frequency",
                "HASH": "K-mer Hashing",
                # "OH": "One-hot"
            }
            selected_methods = [method.strip().upper() for method in args.vectorization.split(',')]
            new_methods = {}
            for method in selected_methods:
                key = code_to_name.get(method)
                if key and key in vectorization_methods:
                    new_methods[key] = vectorization_methods[key]
                else:
                    log_step(f"Warning: Unknown vectorization method '{method}'")
            
            if new_methods:  # Solo actualizar si se encontraron métodos válidos
                vectorization_methods.clear()
                vectorization_methods.update(new_methods)
            else:
                log_step("No valid vectorization methods selected. Using all available methods.")

    # Check that either a sequence file or a directory is provided, but not both
    if args.file and args.directory:
        log_step("Please provide either a sequence file OR a directory, not both.")
        sys.exit(1)

    # Ensure the output and logs directories exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists('logs'):
        os.makedirs('logs')

    try:
        k_range = parse_k_range(args.k)
    except ValueError as e:
        log_step(f"Invalid k range: {e}")
        sys.exit(1)

    # Determine the "input source": a string with the path to a file or directory
    valid_extensions = ('.fasta', '.fna', '.fa', '.fastq')
    if args.file:
        input_source = args.file
        if args.vectorization:  # Ahora esto comprueba si hay una cadena, no un booleano
            if not input_source.endswith('.tsv') or '-mer' not in input_source:
                log_step("In vectorization mode, the input file must be a .tsv file generated with k-mers (containing '-mer' in the name).")
                sys.exit(1)
            check_tsv(input_source)
        else:
            if not input_source.endswith(valid_extensions):
                log_step(f"The file {input_source} does not have a valid extension {valid_extensions}.")
                sys.exit(1)
    elif args.directory:
        input_source = args.directory
        if not os.path.isdir(input_source):
            log_step(f"The provided path ({input_source}) is not a valid directory.")
            sys.exit(1)
    else:
        log_step("Please provide either a sequence file or a directory containing sequence files.")
        sys.exit(1)

    start_total = time.time()
    try:
        # Call the kmers function with the input source (file or directory)
        run_all(args, input_source, k_range)
    except Exception as e:
        log_step(f"Error during k-mer processing: {e}")
        traceback.print_exc()
        log_step("Exiting the program")
        sys.exit(1)
    end_total = time.time()
    log_step(f"Total time: {end_total - start_total} seconds")

if __name__ == "__main__":
    main()
