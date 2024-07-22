import time
import os
import csv
from Bio import SeqIO
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kmer import log_step, save_time_log, log_results, calculate_tfidf, vectorization, time_log, __date__

def kmer_counter(sequence, k):
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1
    return kmer_counts

def evaluate_kmer_vectors(vectors):
    # Example metric: Mean cosine similarity between all pairs of vectors
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix.mean()

def kmers(args, files, k_range):
    best_performance_metric = -float('inf')  # Start with the worst possible metric
    best_k = None
    performance_metrics = []

    for k in k_range:
        kmer_set = set()
        all_sequences = []
        genome_ids = []
        labels = []

        log_step(f"Processing {k}-mer")

        for filepath in files:
            concatenated_sequence = ""
            genome_id = None
            label = os.path.splitext(os.path.basename(filepath))[0]

            for record in SeqIO.parse(filepath, 'fasta'):
                concatenated_sequence += str(record.seq)
                if genome_id is None:
                    genome_id = record.id
            try:
                start_time_vector = time.time()
                kmer_counts = kmer_counter(concatenated_sequence, k)
                elapsed_time_vector = time.time() - start_time_vector

                time_log.append([genome_id, k, elapsed_time_vector])
                all_sequences.append(concatenated_sequence)
                genome_ids.append(genome_id)
                labels.append(label)
                kmer_set.update(kmer_counts.keys())

            except Exception as e:
                log_step(f"Error processing {k}-mer for file {filepath}: {e}")
                save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))
                log_results(args.output_dir, files, best_k, k_range)

            output_filepath = os.path.join(args.output_dir, f'{__date__}_{k}-mer.tsv')
            file_exists = os.path.isfile(output_filepath)
            with open(output_filepath, 'a') as output_file:
                writer = csv.writer(output_file, delimiter='\t')
                sorted_kmers = sorted(kmer_set)
                if not file_exists:
                    writer.writerow(['ID_genome', 'label'] + sorted_kmers)
                writer.writerow([genome_id, label] + [kmer_counts.get(kmer, 0) for kmer in sorted_kmers])

        # Vectorization
        try:
            tfidf_matrix = calculate_tfidf(all_sequences, k)

            # Evaluate the vectors for the current k
            performance_metric = evaluate_kmer_vectors(tfidf_matrix)
            performance_metrics.append((k, performance_metric))
            if performance_metric > best_performance_metric:
                best_performance_metric = performance_metric
                best_k = k

            vector_output_filepath = os.path.join(args.output_dir, f'{__date__}_{k}-vectors.tsv')
            file_exists = os.path.isfile(vector_output_filepath)
            with open(vector_output_filepath, 'a') as output_file:
                writer = csv.writer(output_file, delimiter='\t')
                if not file_exists:
                    writer.writerow(['ID_genome', 'label'] + sorted_kmers)
                for i, genome_id in enumerate(genome_ids):
                    tfidf_vector = tfidf_matrix[i].toarray().flatten()
                    writer.writerow([genome_id, labels[i]] +  tfidf_vector.tolist())

        except Exception as e:
            log_step(f"Error calculating TF-IDF for k={k}: {e}")
            save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))
            log_results(args.output_dir, files, best_k, k_range)

        log_step(f"Completed processing for k={k}, results saved to {output_filepath}")

    log_results(args.output_dir, files, best_k, k_range)
    save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))

    # Save performance metrics to a CSV file for visualization
    metrics_output_filepath = os.path.join(args.output_dir, f'{__date__}_performance_metrics.tsv')
    with open(metrics_output_filepath, 'w') as metrics_file:
        writer = csv.writer(metrics_file, delimiter='\t')
        writer.writerow(['k', 'performance_metric'])
        writer.writerows(performance_metrics)

    # Training and evaluation using the best k
    log_step(f"Training and evaluating model using best k={best_k}")
    best_k_file = os.path.join(args.output_dir, f'{__date__}_{best_k}-vectors.tsv')
    log_step(f"Using vectors from {best_k_file}")
    vectorization(best_k_file)
