import time
import os
import argparse
import traceback
from kmer import log_step, kmers, extract_taxonomy, classification, evaluate_all_vectorizations, save_time_log, __date__
import pandas as pd
import sys

def parse_k_range(k_str):
    if '-' in k_str:
        k_start, k_end = map(int, k_str.split('-'))
        if k_start >= k_end:
            raise ValueError("The first number in the range must be less than the second number")
        return range(k_start, k_end + 1)
    else:
        k = int(k_str)
        return range(k, k + 1)

def check_tsv(file_path):
    """
    Validates if the file is a valid .tsv and meets the requirements.
    """
    # Check if the file has a .tsv extension
    if not file_path.endswith('.tsv'):
        log_step(f"Error: The file {file_path} is not a .tsv file.")
        log_step("Exiting program")
        sys.exit(1)

    try:
        # Read the file
        df = pd.read_csv(file_path, sep='\t')

        # Check if it contains the required columns in order
        required_columns = ['ID_genome', 'label']
        if not all(col in df.columns[:2] for col in required_columns):
            log_step(f"Error: The file {file_path} must contain the required columns {required_columns} in this exact order at the beginning.")
            log_step("Exiting program")
            sys.exit(1)

        # Check if it contains additional columns (k-mers)
        if len(df.columns) <= 2:
            log_step(f"Error: The file {file_path} must contain additional columns corresponding to k-mers.")
            log_step("Exiting program")
            sys.exit(1)

        log_step(f"File {file_path} successfully validated.")
        return df
    except Exception as e:
        log_step(f"Error reading the file {file_path}: {e}")
        log_step("Exiting program")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process sequence files with k-mer vectorization")
    # Grupo mutuamente exclusivo para -k y -v
    # *** ACTUALIZAR: que se puedan utilizar por si solos o todos. e.g., -kvc ***
    # exclusive_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-k', help='Value of k for k-mers (single value or range, e.g., 3 or 2-12)', required=True)
    parser.add_argument('-v', '--vectorization', action='store_true', help='Enable vectorization mode')
    # exclusive_group.add_argument('-c', '--classification', action='store_true', help='Enable classification mode')

    #parser.add_argument('-k', required=True, help='Value of k for k-mers (single value or range, e.g., 3 or 2-12)')
    parser.add_argument('-f', '--file', type=str, help='Path to a sequence file (.fasta, .fna, .fa, .fastq)')
    parser.add_argument('-d', '--directory', type=str, help='Path to a directory containing sequence files')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory for output files')

    args = parser.parse_args()

    if args.file and args.directory:
        log_step("Please provide either a sequence file or a directory containing sequence files, not both")
        return
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists('logs'):
        os.makedirs('logs')

    if args.k:
        valid_extensions = ('.fasta', '.fna', '.fa', '.fastq')

        try:
            k_range = parse_k_range(args.k)
        except ValueError as e:
            log_step(f"Invalid k range: {e}")
            return

        log_step(f"Starting processing with k-mer range {args.k}")
    else:
        valid_extensions = ('.tsv')

    if args.file:
        files = [args.file] if args.file.endswith(valid_extensions) else []
    elif args.directory:
        files = [os.path.join(args.directory, filename) for filename in os.listdir(args.directory) if filename.endswith(valid_extensions)]
    else:
        files = []
        log_step("Please provide a sequence file or a directory containing sequence files")

    if not files:
        log_step("No valid files found to process")
        return

    start_total = time.time()
    try:
        # compare_kmers(args, files, k_range)
        if args.vectorization and not args.k:
            if args.file:
                if '-mer' in args.file:
                    check_tsv(args.file)
                    log_step(f"Starting DNA sequence vectorization and classification process for file f{args.file}")
                    try:
                        df = pd.read_csv(args.file, delimiter='\t')
                        taxonomy_df = df['label'].apply(extract_taxonomy).apply(pd.Series)
                        df['label'] = taxonomy_df['s']
                        df = df.rename(columns={'label': 'specie'})
                        del taxonomy_df
                        results_df = evaluate_all_vectorizations(df)
                        results_output_filepath = os.path.join(args.output_dir, f'{__date__}_k-vectors.tsv')
                        file_exists = os.path.isfile(results_output_filepath)
                        results_df.to_csv(results_output_filepath, sep='\t', index=False)

                    except Exception as e:
                        log_step(f"Error in vectorization for k=k: {e}")
                        save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))
                        log_results(args.output_dir, files, k, k_range)
                        return

            elif args.directory:
                files = [os.path.join(args.directory, filename) for filename in os.listdir(args.directory) if filename.endswith(valid_extensions) and '-mer' in filename]
                for file in files:
                    check_tsv(file)
                    log_step(f"Starting DNA sequence vectorization and classification process for file f{file}")
                    try:
                        df = pd.read_csv(file, delimiter='\t')
                        taxonomy_df = df['label'].apply(extract_taxonomy).apply(pd.Series)
                        df['label'] = taxonomy_df['s']
                        df = df.rename(columns={'label': 'specie'})
                        del taxonomy_df
                        results_df = evaluate_all_vectorizations(df)
                        results_output_filepath = os.path.join(args.output_dir, f'{__date__}_k-vectors.tsv')
                        file_exists = os.path.isfile(results_output_filepath)
                        results_df.to_csv(results_output_filepath, sep='\t', index=False)

                    except Exception as e:
                        log_step(f"Error in vectorization for k=k: {e}")
                        save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))
                        log_results(args.output_dir, files, k, k_range)
                        return

        # elif args.classification:
        #     if args.file:
        #         if '-vectors' in args.file:
        #             check_tsv(args.file)
        #             log_step(f"Starting DNA sequence classification process for file f{args.file}")
        #             classification(pd.read_csv(args.file, delimiter='\t'))
        #     elif args.directory:
        #         files = [os.path.join(args.directory, filename) for filename in os.listdir(args.directory) if filename.endswith(valid_extensions) and '-vectors' in filename]
        #         for file in files:
        #             check_tsv(file)
        #             log_step(f"Starting DNA sequence classification process for file f{file}")
        #             classification(pd.read_csv(file, delimiter='\t'))

        else:
            kmers(args, files, k_range)

    except Exception as e:
        log_step(f"Error during kmer: {e}")
        traceback.print_exc()
        log_step("Exiting program")

        # Delete the files created in output directory in this run
        # for filename in os.listdir(args.output_dir):
        #     if filename.startswith(__date__):
        #         os.remove(os.path.join(args.output_dir, filename))

    end_total = time.time()

    log_step(f"Total time taken: {end_total - start_total} seconds\n")

if __name__ == "__main__":
    main()
