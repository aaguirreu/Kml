import time
import os
import csv
import psutil
from Bio import SeqIO
from Bio.Seq import Seq
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed
from kmer import log_step, save_time_log, log_results, evaluate_all_vectorizations, classification, time_log, __date__

def kmer_counter(sequence, k):
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        rev_kmer = str(Seq(kmer).reverse_complement())
        min_kmer = min(kmer, rev_kmer)
        if min_kmer in kmer_counts:
            kmer_counts[min_kmer] += 1
        else:
            kmer_counts[min_kmer] = 1
    return kmer_counts

def determine_num_chunks(sequence_length, max_chunks=None):
    num_cores = os.cpu_count() or 1
    estimated_chunks = min(num_cores, sequence_length // 1000)
    if max_chunks is not None:
        estimated_chunks = min(estimated_chunks, max_chunks)
    return max(1, estimated_chunks)

def parallel_kmer_counter(sequence, k):
    num_chunks = determine_num_chunks(len(sequence))
    chunk_size = len(sequence) // num_chunks
    chunks = [sequence[i:i + chunk_size + k - 1] for i in range(0, len(sequence), chunk_size)]
    
    kmer_counts = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(kmer_counter, chunk, k) for chunk in chunks]
        
        for future in as_completed(futures):
            result = future.result()
            for kmer, count in result.items():
                if kmer in kmer_counts:
                    kmer_counts[kmer] += count
                else:
                    kmer_counts[kmer] = count
                    
    return kmer_counts

def clean_sequence(sequence):
    valid_bases = set("ACGT")
    return ''.join(base for base in sequence if base in valid_bases)

def evaluate_kmer_vectors(vectors):
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix.mean()

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    # Log de la memoria utilizada en MB
    log_step(f"Uso de memoria: {memory_info.rss / (1024 ** 2):.2f} MB")

def extract_taxonomy(label):
    if '[' in label:
        label = label.split('[')[0] 
    if ';' in label:
        parts = label.split(';')
    elif ',' in label:
        label = label.split(',')[0]
        parts = label.split(' ')
        if ':' in label:
            label = label.split(':')[1].strip()
            parts = label.split(' ')
    else:
        parts = label.split(' ')
    taxonomy = {}
    if '__' in label:
        for part in parts:
            level, value = part.split('__') if '__' in part else (part, "")
            taxonomy[level] = value
    else:
        taxonomy['s'] = ' '.join(parts[:2])
    return taxonomy

def kmers(args, files, k_range):
    best_performance_metric = -float('inf')  # Start with the worst possible metric
    best_k = None
    performance_metrics = []

    for k in k_range:
        kmer_set = set()
        data = []

        log_step(f"Processing {k}-mer")
        for filepath in files:
            for record in SeqIO.parse(filepath, 'fasta'):
                label = record.description.replace(record.id, "").strip()
                sequence = str(record.seq)
                try:
                    cleaned_sequence = clean_sequence(sequence)
                    # if len(cleaned_sequence) < len(sequence):
                    #     log_step("Found invalid characters in sequence. Characters differente from ATCG were deleted")
                
                    start_time_vector = time.time()
                    kmer_counts = parallel_kmer_counter(cleaned_sequence, k)
                    elapsed_time_vector = time.time() - start_time_vector
                    time_log.append([record.id, k, elapsed_time_vector])
                    data.append({
                        'ID_genome': record.id,
                        'label': label,
                        **kmer_counts
                    })
                    kmer_set.update(kmer_counts.keys())

                except Exception as e:
                    log_step(f"Error processing {k}-mer for file {filepath}: {e}")
                    save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))
                    log_results(args.output_dir, files, best_k, k_range)
                    return

            log_step(f"reading {filepath}")
            log_memory_usage()

        log_step(f"Creating df from data")
        df = pd.DataFrame(data)
        log_memory_usage()
        df.fillna(0, inplace=True)
        sorted_cols = sorted(df.columns.difference(['ID_genome', 'label']))
        new_order = ['ID_genome', 'label'] + sorted_cols
        log_step(f"Ordering df k-mers A-Z")
        df = df[new_order]
        log_memory_usage()

        output_filepath = os.path.join(args.output_dir, f'{__date__}_{k}-mer.tsv')
        file_exists = os.path.isfile(output_filepath)
        df.to_csv(output_filepath, sep='\t', index=False)
        
        # Vectorization and Machine Learning
        try:
            taxonomy_df = df['label'].apply(extract_taxonomy).apply(pd.Series)
            df['label'] = taxonomy_df['s']
            df = df.rename(columns={'label': 'specie'})
            del taxonomy_df
            log_memory_usage()
            results = evaluate_all_vectorizations(df)

            # save vectors in .tsv?
            # vector_output_filepath = os.path.join(args.output_dir, f'{__date__}_{k}-vectors.tsv')
            # file_exists = os.path.isfile(vector_output_filepath)
            # vector_df.to_csv(vector_output_filepath, sep='\t', index=False)

        except Exception as e:
            log_step(f"Error in vectorization for k={k}: {e}")
            save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))
            log_results(args.output_dir, files, k, k_range)
            return

        log_step(f"Completed processing for k={k}, results saved to {output_filepath}")
        # Training and evaluation using the best k
        # log_step(f"Training and evaluating model using k={k}")
        # log_step(f"Using vectors from {vector_output_filepath}")
        # classification(vector_df)

    log_results(args.output_dir, files, best_k, k_range)
    save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))

    # Save performance metrics to a CSV file for visualization
    metrics_output_filepath = os.path.join(args.output_dir, f'{__date__}_performance_metrics.tsv')
    with open(metrics_output_filepath, 'w') as metrics_file:
        writer = csv.writer(metrics_file, delimiter='\t')
        writer.writerow(['k', 'performance_metric'])
        writer.writerows(performance_metrics)
