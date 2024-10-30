import time
import os
import argparse
from kmer import log_step, kmers, classification, __date__
import pandas as pd

def parse_k_range(k_str):
    if '-' in k_str:
        k_start, k_end = map(int, k_str.split('-'))
        if k_start >= k_end:
            raise ValueError("The first number in the range must be less than the second number")
        return range(k_start, k_end + 1)
    else:
        k = int(k_str)
        return range(k, k + 1)

def main():
    parser = argparse.ArgumentParser(description="Process sequence files with k-mer vectorization")
    # Grupo mutuamente exclusivo para -k y -v
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument('-k', help='Value of k for k-mers (single value or range, e.g., 3 or 2-12)')
    exclusive_group.add_argument('-c', '--classification', action='store_true', help='Enable classification mode')

    #parser.add_argument('-k', required=True, help='Value of k for k-mers (single value or range, e.g., 3 or 2-12)')
    parser.add_argument('-f', '--file', type=str, help='Path to a sequence file (.fasta, .fna, .fa, .fastq)')
    parser.add_argument('-d', '--directory', type=str, help='Path to a directory containing sequence files')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory for output files')

    args = parser.parse_args()

    if args.file and args.directory:
        print("Please provide either a sequence file or a directory containing sequence files, not both")
        return
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists('logs'):
        os.makedirs('logs')

    if args.classification:
        valid_extensions = ('.tsv')
    else:
        valid_extensions = ('.fasta', '.fna', '.fa', '.fastq')

        try:
            k_range = parse_k_range(args.k)
        except ValueError as e:
            print(f"Invalid k range: {e}")
            return

        log_step(f"Starting processing with k-mer range {args.k}")

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
    # try:
        # compare_kmers(args, files, k_range)
    if args.classification:
        if args.file:
            log_step(f"Starting DNA sequence vectorization process for file f{args.file}")
            classification(pd.read_csv(args.file, delimiter='\t'))
        elif args.directory:
            files = [os.path.join(args.directory, filename) for filename in os.listdir(args.directory) if filename.endswith(valid_extensions) and 'vectors' in filename]
            for file in files:
                log_step(f"Starting DNA sequence vectorization process for file f{file}")
                classification(pd.read_csv(file, delimiter='\t'))

    else:
        kmers(args, files, k_range)

    # except Exception as e:
    #     log_step(f"Error during kmer: {e}")
    #     log_step("Exiting program")

        # Delete the files created in output directory in this run
        # for filename in os.listdir(args.output_dir):
        #     if filename.startswith(__date__):
        #         os.remove(os.path.join(args.output_dir, filename))

    end_total = time.time()

    log_step(f"Total time taken: {end_total - start_total} seconds\n")

if __name__ == "__main__":
    main()
