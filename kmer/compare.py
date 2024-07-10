import time
import os
import csv
from Bio import SeqIO
import itertools
from kmer import log_step, save_time_log, log_results, step_log, time_log, __date__
#from kmer import sec2vec

def vector(sequence, k):
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1
    return kmer_counts

def all_posible_kmers(k):
    bases = ['A', 'T', 'C', 'G']
    return sorted([''.join(p) for p in itertools.product(bases, repeat=k)])

def compare_kmers(args, files, k_range):
    best_performance_file = None
    best_performance_time = float('inf')
    best_k = None

    for k in k_range:
        kmer_list = all_posible_kmers(k)
        kmer_set = set()

        log_step(f"Processing {k}-mer")

        for filepath in files:
            concatenated_sequence = ""
            genome_id = None
            for record in SeqIO.parse(filepath, 'fasta'):
                concatenated_sequence += str(record.seq)
                if genome_id is None:
                    genome_id = record.id
            try:
                start_time_vector = time.time()
                result_py = vector(concatenated_sequence, k)
                elapsed_time_vector = time.time() - start_time_vector

                start_time_sec2vec = time.time()
                #result_cpp = sec2vec(concatenated_sequence, k)
                result_cpp = vector(concatenated_sequence, k)
                elapsed_time_sec2vec = time.time() - start_time_sec2vec

                if result_cpp != result_py:
                    log_step(f"Results do not match for file {filepath}")

                time_log.append([genome_id, k, elapsed_time_vector, elapsed_time_sec2vec])
                kmer_set.update(result_py.keys())

                if elapsed_time_vector < best_performance_time:
                    best_performance_time = elapsed_time_vector
                    best_k = k

            except Exception as e:
                log_step(f"Error processing {k}-mer for file {filepath}: {e}")
                save_time_log(os.path.join(args.output_dir, f'{__date__}.tab'))
                log_results([filepath], filepath, best_k, k_range)

            output_filepath = os.path.join(args.output_dir, f'{__date__}_{k}-mer.tab')
            file_exists = os.path.isfile(output_filepath)
            with open(output_filepath, 'a') as output_file:
                writer = csv.writer(output_file, delimiter='\t')
                if not file_exists:
                    writer.writerow(['ID_genome'] + kmer_list)
                writer.writerow([genome_id] + [result_py.get(kmer, 0) for kmer in kmer_list])

        log_step(f"Completed processing for k={k}, results saved to {output_filepath}")
    
    log_results(files, best_performance_file, best_k, k_range)
    save_time_log(os.path.join(args.output_dir, f'{__date__}.tab'))