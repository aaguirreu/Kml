from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from kmer import log_step, run_models
import hashlib
import pandas as pd
import numpy as np
import psutil

def vectorize_kmer_frequency(df):
    return df

def vectorize_tfidf(df):
    """
    Perform a tf-df vectorization on k-mer columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing k-mers.
    
    Returns:
        pd.DataFrame: A DataFrame with the tf-idf k-mer values.
    """
    specie_column = df["specie"]
    kmer_data = df.drop(columns=["ID_genome", "specie"])
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(kmer_data.astype(float))
    vectors_df = pd.DataFrame(tfidf_matrix.toarray(), columns=kmer_data.columns, index=df["ID_genome"])
    vectors_df = vectors_df.reset_index().rename(columns={"index": "ID_genome"})
    vectors_df.insert(1, 'specie', specie_column)
    return vectors_df

def vectorize_relative_frequency(df):
    """
    Perform a relative frequency vectorization on k-mer columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing k-mers.
    
    Returns:
        pd.DataFrame: A DataFrame with the relative frequency k-mer values.
    """
    vector_list = []
    for index, row in df.iterrows():
        total_kmers = row[2:].sum()
        if total_kmers > 0:
            normalized = row[2:] / total_kmers 
        else:
            normalized = row[2:]
        vector_list.append(normalized)
    vectors_df = pd.DataFrame(vector_list, columns=df.columns[2:])
    vectors_df.insert(0, 'ID_genome', df['ID_genome'])
    vectors_df.insert(1, 'specie', df['specie'])
    vector_list
    return vectors_df

def vectorize_kmer_hashing(df):
    """
    Perform k-mer hashing vectorization on k-mer columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing k-mers.
    
    Returns:
        pd.DataFrame: A DataFrame with k-mer hashing k-mer values.
    """
    hashed_vectors = []
    num_buckets = 10
    for _, row in df.iterrows():
        vector = [0] * num_buckets
        for kmer, count in row[2:].items():
            bucket_index = int(hashlib.sha256(kmer.encode()).hexdigest(), 16) % num_buckets
            vector[bucket_index] += count 
        hashed_vectors.append(vector)
    vectors_df = pd.DataFrame(hashed_vectors, columns=[f'bucket_{i}' for i in range(num_buckets)])
    vectors_df.insert(0, 'ID_genome', df['ID_genome'])
    vectors_df.insert(1, 'specie', df['specie'])
    return vectors_df

def vectorize_onehot(df):
    """
    Perform one-hot encoding on k-mer columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing k-mers.
    
    Returns:
        pd.DataFrame: A DataFrame with one-hot encoded k-mer values.
    """

    # Separate ID_genome and label columns
    id_label_columns = ['ID_genome', 'label']
    if not all(col in df.columns for col in id_label_columns):
        raise ValueError("Both 'ID_genome' and 'label' columns must be present.")
    
    # Identify k-mer columns (all columns after 'label')
    kmer_columns = df.columns[2:]  # Assumes 'ID_genome' and 'label' are the first two columns
    df = df.head()
    # Perform one-hot encoding
    one_hot_encoded_kmers = pd.get_dummies(df[kmer_columns].astype(str), prefix=kmer_columns)
    
    # Concatenate ID_genome, label, and the one-hot encoded k-mers
    result_df = pd.concat([df[id_label_columns], one_hot_encoded_kmers], axis=1)
    
    return result_df

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    # Log de la memoria utilizada en MB
    log_step(f"Uso de memoria: {memory_info.rss / (1024 ** 2):.2f} MB")

vectorization_methods = {
    'K-mer Frequency': vectorize_kmer_frequency,
    # 'TF-IDF': vectorize_tfidf,
    # 'Relative Frequency': vectorize_relative_frequency,
    # 'K-mer Hashing': vectorize_kmer_hashing,
}

def evaluate_all_vectorizations(df):
    """
    Recorre los métodos de vectorización y llama a `run_models` para cada uno.
    
    Args:
        file_path (str): Ruta al archivo con los datos de entrada.
    
    Returns:
        pd.DataFrame: Tabla con los resultados para cada vectorización y modelo.
    """
    results = []
    for vectorization_name, vectorization_function in vectorization_methods.items():
        log_step(f"Applying vectorization: {vectorization_name}")
        log_memory_usage()
        vectorized_data = vectorization_function(df)
        X_train, X_test, y_train, y_test, model_results = run_models(vectorized_data)
        for model_name, metrics in model_results.items():
            results.append({
                'Vectorization': vectorization_name,
                'Model': model_name,
                **metrics
            })
    results_df = pd.DataFrame(results)
    return results_df