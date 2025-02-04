import time
import os
import csv
import psutil
import traceback
from Bio import SeqIO
from Bio.Seq import Seq
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed
from kmer import log_step, save_time_log, log_results, evaluate_all_vectorizations, classification, time_log, __date__
import cudf
import re

# def kmer_counter(sequence, k):
#     kmer_counts = {}
#     for i in range(len(sequence) - k + 1):
#         kmer = sequence[i:i + k]
#         rev_kmer = str(Seq(kmer).reverse_complement())
#         min_kmer = min(kmer, rev_kmer)
#         if min_kmer in kmer_counts:
#             kmer_counts[min_kmer] += 1
#         else:
#             kmer_counts[min_kmer] = 1
#     return kmer_counts

def encode_kmer(kmer):
    """Converts a k-mer from characters to its numerical representation."""

    encoding = {'C': 0, 'A': 1, 'T': 2, 'G': 3}
    return sum(encoding[base] * (4 ** i) for i, base in enumerate(reversed(kmer)))

def decode_kmer(value, k):
    """Converts a numerical value back to a k-mer."""
    decoding = ['C', 'A', 'T', 'G']
    kmer = []
    for _ in range(k):
        kmer.append(decoding[value % 4])
        value //= 4
    return ''.join(reversed(kmer))

def kmer_counter(sequence, k):
    """Cuenta los k-mers válidos en una secuencia."""
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if is_valid_kmer(kmer):  # Validar k-mer directamente
            rev_kmer = str(Seq(kmer).reverse_complement())
            kmer_encoded = encode_kmer(kmer)
            rev_kmer_encoded = encode_kmer(rev_kmer)
            min_kmer_encoded = min(kmer_encoded, rev_kmer_encoded)
            if min_kmer_encoded in kmer_counts:
                kmer_counts[min_kmer_encoded] += 1
            else:
                kmer_counts[min_kmer_encoded] = 1
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

def is_valid_kmer(kmer):
    """Valida si un k-mer contiene únicamente las bases 'A', 'C', 'T', 'G'."""
    return all(base in "ACTG" for base in kmer)

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

def create_dataframe_chunk(data, start_idx, chunk_size):
    return pd.DataFrame(data[start_idx:start_idx + chunk_size])

def process_chunk(args):
    data, start_idx, chunk_size = args
    return create_dataframe_chunk(data, start_idx, chunk_size)

def process_kmer(args, files, k, best_k):
    kmer_set = set()
    data = []

    log_step(f"Processing {k}-mer")
    genome_kmer_counts = {}

    for filepath in files:
        log_step(f"Reading file: {filepath}")
        for record in SeqIO.parse(filepath, 'fasta'):
            label = record.description.replace(record.id, "").strip()
            sequence = str(record.seq)
            try:
                start_time_vector = time.time()
                kmer_counts = parallel_kmer_counter(sequence, k)
                elapsed_time_vector = time.time() - start_time_vector
                time_log.append([record.id, k, elapsed_time_vector])

                # Extraer la parte del ID que comienza con letras y caracteres especiales, incluyendo ceros y el número que sigue al último cero
                base_id = re.match(r'^[^\d]+0*\d{1}', record.id).group()

                if base_id not in genome_kmer_counts:
                    genome_kmer_counts[base_id] = kmer_counts
                else:
                    for kmer, count in kmer_counts.items():
                        if kmer in genome_kmer_counts[base_id]:
                            genome_kmer_counts[base_id][kmer] += count
                        else:
                            genome_kmer_counts[base_id][kmer] = count

                existing_entry = next((item for item in data if item['ID_genome'] == base_id), None)
                if existing_entry:
                    for kmer, count in genome_kmer_counts[base_id].items():
                        if kmer in existing_entry:
                            existing_entry[kmer] += count
                        else:
                            existing_entry[kmer] = count
                else:
                    data.append({
                        'ID_genome': base_id,
                        'label': label,
                        **genome_kmer_counts[base_id]
                    })

                kmer_set.update(genome_kmer_counts[base_id].keys())

            except Exception as e:
                log_step(f"Error processing {k}-mer for file {filepath}: {e}")
                save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))
                log_results(args.output_dir, files, best_k, k_range)
                return

        log_step("File read successfully")
        log_memory_usage()

    try:
        log_step(f"Creating DF from data")
        # Aquí se eliminó la creación del DataFrame cuDF
        log_memory_usage()

        log_step(f"Ordering df k-mers C-A-T-G")
        # Asumiendo que data es una lista de diccionarios
        sorted_cols = list(set().union(*(d.keys() for d in data)) - {'ID_genome', 'label'})

        # Decodificar las columnas utilizando operaciones vectorizadas
        # Supongamos que decode_kmer puede ser vectorizada
        decoded_column_names = [decode_kmer(col, k) for col in sorted_cols]

        # Crear un mapeo de columnas originales a columnas decodificadas
        column_mapping = dict(zip(sorted_cols, decoded_column_names))

        # Renombrar las columnas en los datos originales
        for d in data:
            for old_col, new_col in column_mapping.items():
                if old_col in d:
                    d[new_col] = d.pop(old_col)
        
        # Asegurarse de que 'ID_genome' y 'label' estén al comienzo
        all_keys = ['ID_genome', 'label'] + decoded_column_names

    except Exception as e:
        log_step(f"Error creating df from data for k={k}: {e}")
        traceback.print_exc()
        save_time_log(os.path.join(args.output_dir, f'{__date__}.tsv'))
        log_results(args.output_dir, files, k, k_range)
        return

    # output_filepath = os.path.join(args.output_dir, f'{__date__}_{k}-mer.tsv')
    # file_exists = os.path.isfile(output_filepath)
    # df.to_csv(output_filepath, sep='\t', index=False)
    return all_keys, data

def kmers(args, files, k_range):
    best_performance_metric = -float('inf')  # Start with the worst possible metric
    best_k = None
    performance_metrics = []
    for k in k_range:
        all_keys, data = process_kmer(args, files, k, best_k)
        # if not args.vectorization:
        #     del data
        # # Vectorization and Machine Learning
        # else:
        if args.vectorization:
            try:
                # CAMBIAR DF A DATA; TRATAR COMO NUMPY ARRAY
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
    
        output_filepath = os.path.join(args.output_dir, f'{__date__}_{k}-mer.tsv')
        log_step(f"Completed processing for k={k}")
        try:
            with open(output_filepath, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=all_keys, delimiter='\t')
                writer.writeheader()
                for row in data:
                    # Asegurarse de que cada diccionario tenga todas las claves
                    writer.writerow({key: row.get(key, '') for key in all_keys})
            log_step(f"Data saved to CSV at {output_filepath}")
        except ValueError as ve:
            log_step(f"Error writing CSV for k={k}: {ve}")
        except Exception as e:
            log_step(f"Unexpected error writing CSV for k={k}: {e}")

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
