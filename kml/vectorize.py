import hashlib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder

def vectorize_kmer_frequency(df):
    """
    In this method no transformation is applied,
    assuming the DataFrame already contains the absolute k-mer counts.
    """
    return df

def vectorize_tfidf(df):
    """
    Performs TF-IDF vectorization on the k-mer columns in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing:
            - "file"
            - "specie" (or "specie")
            - All other columns are k-mer counts.
    
    Returns:
        pd.DataFrame: DataFrame with "file", "specie", and the transformed k-mer values.
    """
    specie_column = df["specie"]
    kmer_data = df.drop(columns=["file", "specie"])
    transformer = TfidfTransformer()
    # Convertir a float para TF-IDF
    tfidf_matrix = transformer.fit_transform(kmer_data.astype(float))
    vectors_df = pd.DataFrame(tfidf_matrix.toarray(), columns=kmer_data.columns, index=df["file"])
    vectors_df = vectors_df.reset_index().rename(columns={"index": "file"})
    vectors_df.insert(1, "specie", specie_column.values)
    return vectors_df

def vectorize_relative_frequency(df):
    vector_list = []
    for index, row in df.iterrows():
        total_kmers = row[2:].sum()
        if total_kmers > 0:
            normalized = row[2:] / total_kmers 
        else:
            normalized = row[2:]
        
        vector_list.append(normalized)

    vectors_df = pd.DataFrame(vector_list, columns=df.columns[2:])

    vectors_df.insert(0, 'file', df['file'])
    vectors_df.insert(1, 'specie', df['specie'])
    vector_list
    return vectors_df

def vectorize_kmer_hashing(df, num_buckets=10):
    """
    Performs k-mer hashing vectorization.
    
    For each k-mer in each row, a bucket is computed (using SHA256) and its count is added to that bucket.
    
    Parameters:
        df (pd.DataFrame): DataFrame with "file", "specie", and k-mer counts.
        num_buckets (int): Number of buckets to use.
    
    Returns:
        pd.DataFrame: DataFrame with "file", "specie", and bucket columns (e.g., bucket_0, bucket_1, ...).
    """
    hashed_vectors = []
    for _, row in df.iterrows():
        vector = [0] * num_buckets
        for kmer, count in row[2:].items():
            bucket_index = int(hashlib.sha256(kmer.encode()).hexdigest(), 16) % num_buckets
            vector[bucket_index] += count
        hashed_vectors.append(vector)
    vectors_df = pd.DataFrame(hashed_vectors, columns=[f"bucket_{i}" for i in range(num_buckets)])
    vectors_df.insert(0, "file", df["file"].values)
    vectors_df.insert(1, "specie", df["specie"].values)
    return vectors_df

def vectorize_onehot(df):
    """
    Performs one-hot encoding on the k-mer columns in the DataFrame using sklearn's OneHotEncoder.
    
    Converts counts to binary (presence/absence) and uses drop='if_binary'
    to avoid duplicating binary features.
    """
    id_specie_columns = ['file', 'specie']
    if not all(col in df.columns for col in id_specie_columns):
        raise ValueError("Both 'file' and 'specie' columns must be present.")
    
    kmer_data = df.drop(columns=id_specie_columns)
    kmer_presence = (kmer_data > 0).astype(int)
    
    # Use drop='if_binary' to avoid duplicating binary features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary')
    encoded_data = encoder.fit_transform(kmer_presence)
    
    # Generate feature names using the original k-mer columns
    feature_names = [f"{col}_present" for col in kmer_data.columns]
    
    vectors_df = pd.DataFrame(encoded_data, columns=feature_names)
    vectors_df.insert(0, 'file', df['file'])
    vectors_df.insert(1, 'specie', df['specie'])
    
    return vectors_df

# Dictionary containing the vectorization methods to evaluate
vectorization_methods = {
    'K-mer Frequency': vectorize_kmer_frequency,
    'TF-IDF': vectorize_tfidf,
    'Relative Frequency': vectorize_relative_frequency,
    # 'K-mer Hashing': vectorize_kmer_hashing,
    # 'One-hot': vectorize_onehot,
}