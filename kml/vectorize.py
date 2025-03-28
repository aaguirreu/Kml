import hashlib
import polars as pl
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import warnings
import gc
import psutil

def time_vectorization(func):
    """Decorator to measure execution time of vectorization functions"""
    def wrapper(df, *args, **kwargs):
        start_time = time.time()
        result = func(df, *args, **kwargs)
        
        # Force evaluation of the DataFrame to get accurate timing
        try:
            df_shape = result.shape  # This forces execution of lazy operations
        except Exception as e:
            print(f"Warning: Error forcing evaluation in time_vectorization: {e}")
            
        execution_time = time.time() - start_time
        # Add execution time as metadata to the DataFrame (using a custom attribute)
        result._vectorization_time = execution_time
        return result
    return wrapper

@time_vectorization
def vectorize_kmer_frequency(df):
    """
    In this method no transformation is applied,
    assuming the DataFrame already contains the absolute k-mer counts.
    """
    return df

@time_vectorization
def vectorize_tfidf(df: pl.DataFrame) -> pl.DataFrame:
    """
    Performs the TF-IDF transformation on the k-mer columns.
    
    TF (Term Frequency): each value is divided by the sum of the row.
    IDF (Inverse Document Frequency): for each k-mer, it calculates 
        log((n_rows + 1) / (doc_freq + 1)) + 1,
    where doc_freq is the number of rows where the k-mer is greater than 0.
    
    Parameters:
        df (pl.DataFrame): DataFrame with columns "file", "specie" and k-mer counts.
        
    Returns:
        pl.DataFrame: DataFrame with columns "file", "specie" and TF-IDF values for each k-mer.
    """
    # Extract the "specie" and "file" columns
    specie_column = df["specie"]
    file_column = df["file"]
    
    # Select the k-mer columns
    kmer_data = df.drop(["file", "specie"])
    
    # Convert the k-mers to float and extract as a NumPy array
    kmer_np = kmer_data.with_columns([pl.col(c).cast(pl.Float64) for c in kmer_data.columns]).to_numpy()
    
    # Create and apply the TF-IDF transformer
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(kmer_np)
    
    # Convert the TF-IDF matrix to a dense array
    dense_tfidf = tfidf_matrix.toarray()
    
    # Create a Polars DataFrame with the original k-mer column names
    vectors_df = pl.DataFrame(dense_tfidf, schema=kmer_data.columns)
    
    # Insert the "file" and "specie" columns at the end and then reorder them
    vectors_df = vectors_df.with_columns([
        pl.Series("file", file_column.to_numpy()),
        pl.Series("specie", specie_column.to_numpy())
    ])
    vectors_df = vectors_df.select(["file", "specie"] + kmer_data.columns)
    
    return vectors_df

@time_vectorization
def vectorize_relative_frequency(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the relative frequency of each k-mer in each sample.

    For each row, the total k-mers are summed and each value is divided by that sum.

    Parameters:
        df (pl.DataFrame): DataFrame with columns "file", "specie" and k-mer counts.

    Returns:
        pl.DataFrame: DataFrame with columns "file", "specie" and the k-mers converted
                      to relative frequencies.
    """
    id_cols = ["file", "specie"]
    numeric_cols = [col for col in df.columns if col not in id_cols]

    # Calculate the horizontal sum of k-mers in each row.
    df = df.with_columns(pl.sum_horizontal(numeric_cols).alias("row_sum"))

    # Divide all numeric columns by 'row_sum' in a single operation.
    # Avoid divisions by zero by adding a small epsilon
    df = df.with_columns(
        [(pl.col(col) / (pl.col("row_sum") + 1e-10)).alias(col) for col in numeric_cols]
    ).drop("row_sum")

    return df

@time_vectorization
def vectorize_kmer_hashing(df: pl.DataFrame, num_buckets: int = 10) -> pl.DataFrame:
    """
    Performs vectorization by hashing k-mers.
    
    For each k-mer column, a bucket is calculated (using SHA256) and the counts 
    of all k-mers that fall into the same bucket are aggregated.
    
    This implementation iterates row by row (which may be less efficient for very large datasets).
    
    Parameters:
        df (pl.DataFrame): DataFrame with columns "file", "specie" and k-mer counts.
        num_buckets (int): Number of buckets to use.
    
    Returns:
        pl.DataFrame: DataFrame with columns "file", "specie" and buckets (e.g., bucket_0, bucket_1, ...).
    """
    id_cols = ["file", "specie"]
    numeric_cols = [col for col in df.columns if col not in id_cols]
    
    # Precalculate the assignment of each k-mer to a bucket
    bucket_map = {col: int(hashlib.sha256(col.encode()).hexdigest(), 16) % num_buckets for col in numeric_cols}
    
    # Function to be applied to each row to group counts by bucket
    def hash_row(row: dict) -> list:
        buckets = [0] * num_buckets
        for col in numeric_cols:
            buckets[bucket_map[col]] += row[col]
        return buckets

    # Apply the hashing function row by row; the result is a column of lists
    df = df.with_columns(
        pl.struct(numeric_cols)
        .apply(lambda s: hash_row(s), return_dtype=pl.List(pl.Int64))
        .alias("hashed")
    )
    
    # Expand the column of lists into separate columns for each bucket
    for i in range(num_buckets):
        df = df.with_columns(pl.col("hashed").arr.get(i).alias(f"bucket_{i}"))
    
    df = df.drop("hashed")
    return df

@time_vectorization
def vectorize_onehot(df: pl.DataFrame) -> pl.DataFrame:
    """
    Performs one-hot encoding on the k-mer columns.
    
    Converts counts to binary values (1 if the k-mer is present, 0 if not).
    
    Parameters:
        df (pl.DataFrame): DataFrame with columns "file", "specie" and k-mer counts.
        
    Returns:
        pl.DataFrame: DataFrame with columns "file", "specie" and the k-mers encoded in binary.
    """
    id_cols = ["file", "specie"]
    numeric_cols = [col for col in df.columns if col not in id_cols]
    
    for col in numeric_cols:
        df = df.with_columns((pl.col(col) > 0).cast(pl.Int64).alias(col))
    
    return df

@time_vectorization
def vectorize_tfidf_memory_efficient(df: pl.DataFrame, batch_size=1000) -> pl.DataFrame:
    """
    Memory-efficient version of TF-IDF transformation that processes data in batches
    and uses sparse matrices for large k-mer sets.
    
    Parameters:
        df (pl.DataFrame): DataFrame with columns "file", "specie" and k-mer counts.
        batch_size (int): Number of rows to process in each batch.
        
    Returns:
        pl.DataFrame: DataFrame with columns "file", "specie" and TF-IDF values for each k-mer.
    """
    # Extract the "specie" and "file" columns
    specie_column = df["specie"]
    file_column = df["file"]
    
    # Select the k-mer columns
    kmer_data = df.drop(["file", "specie"])
    
    # Check available memory before processing
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)  # Available memory in GB
    
    # Estimate memory required (rough calculation)
    n_samples = df.shape[0]
    n_features = len(kmer_data.columns)
    estimated_dense_gb = (n_samples * n_features * 8) / (1024 ** 3)  # 8 bytes per float64
    
    # Log memory usage information
    print(f"Available memory: {available_gb:.2f} GB")
    print(f"Estimated memory for dense matrix: {estimated_dense_gb:.2f} GB")
    
    # Warning if memory might be an issue
    if estimated_dense_gb > available_gb * 0.7:  # If estimated to use >70% of available memory
        warnings.warn(
            f"Potential memory issue detected: The operation might require {estimated_dense_gb:.2f} GB "
            f"but only {available_gb:.2f} GB is available. Using sparse matrices and batch processing."
        )
    
    # Create sparse matrix from DataFrame
    rows = []
    cols = []
    data = []
    
    # Process in batches to reduce memory usage
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = kmer_data[start_idx:end_idx]
        
        # Convert batch to numpy for processing
        for i, row_idx in enumerate(range(start_idx, end_idx)):
            row_values = batch.row(i)
            for j, val in enumerate(row_values):
                if val > 0:  # Only store non-zero values
                    rows.append(row_idx)
                    cols.append(j)
                    data.append(val)
        
        # Force garbage collection after each batch
        del batch
        gc.collect()
    
    # Create sparse CSR matrix
    sparse_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))
    
    # Apply TF-IDF transformation
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(sparse_matrix)
    
    # Convert back to DataFrame in batches
    result_chunks = []
    schema = kmer_data.columns
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_tfidf = tfidf_matrix[start_idx:end_idx].toarray()
        
        # Create chunk DataFrame
        chunk_df = pl.DataFrame(batch_tfidf, schema=schema)
        
        # Add file and specie columns
        chunk_df = chunk_df.with_columns([
            pl.Series("file", file_column[start_idx:end_idx].to_numpy()),
            pl.Series("specie", specie_column[start_idx:end_idx].to_numpy())
        ])
        
        result_chunks.append(chunk_df)
        
        # Force garbage collection
        del batch_tfidf
        del chunk_df
        gc.collect()
    
    # Combine chunks
    result_df = pl.concat(result_chunks)
    
    # Reorder columns to match expected format
    result_df = result_df.select(["file", "specie"] + schema)
    
    return result_df

@time_vectorization
def vectorize_relative_frequency_memory_efficient(df: pl.DataFrame, batch_size=1000) -> pl.DataFrame:
    """
    Memory-efficient version that calculates the relative frequency of each k-mer
    in smaller batches to reduce memory usage.
    
    Parameters:
        df (pl.DataFrame): DataFrame with columns "file", "specie" and k-mer counts.
        batch_size (int): Number of rows to process in each batch.
        
    Returns:
        pl.DataFrame: DataFrame with columns "file", "specie" and relative frequencies.
    """
    # Extract metadata columns
    id_cols = ["file", "specie"]
    file_col = df["file"]
    specie_col = df["specie"]
    
    # Get numeric columns
    numeric_cols = [col for col in df.columns if col not in id_cols]
    
    # Process in batches
    result_chunks = []
    n_samples = df.shape[0]
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = df[start_idx:end_idx]
        
        # Calculate row sums
        batch = batch.with_columns(pl.sum_horizontal(numeric_cols).alias("row_sum"))
        
        # Compute relative frequencies
        batch = batch.with_columns(
            [(pl.col(col) / (pl.col("row_sum") + 1e-10)).alias(col) for col in numeric_cols]
        ).drop("row_sum")
        
        result_chunks.append(batch)
        
        # Force garbage collection
        del batch
        gc.collect()
    
    # Combine chunks
    result_df = pl.concat(result_chunks)
    
    return result_df

# Update the dictionary with memory-efficient versions
vectorization_methods = {
    'K-mer Frequency': vectorize_kmer_frequency,
    'TF-IDF': vectorize_tfidf_memory_efficient,  # Use memory-efficient version by default
    'Relative Frequency': vectorize_relative_frequency_memory_efficient,  # Use memory-efficient version by default
    # 'K-mer Hashing': vectorize_kmer_hashing,
    # 'One-hot': vectorize_onehot,
}

# PCA is now a utility function, not a vectorization method
def apply_pca(df: pl.DataFrame, n_components: int = None, prefix: str = "") -> pl.DataFrame:
    """
    Applies Principal Component Analysis (PCA) to the feature columns.
    Memory-optimized version that processes data in batches when needed.
    
    Parameters:
        df (pl.DataFrame): DataFrame with columns "file", "specie" and feature columns.
        n_components (int, optional): Number of principal components to keep.
                                     If None, uses 90% of the number of feature columns.
        prefix (str): Optional prefix for the vectorization method in the name.
        
    Returns:
        pl.DataFrame: DataFrame with columns "file", "specie" and PCA components.
    """
    from sklearn.decomposition import PCA
    import psutil
    
    start_time = time.time()  # Start timing for PCA
    
    # Extract the "specie" and "file" columns
    specie_column = df["specie"]
    file_column = df["file"]
    
    # Select the feature columns
    feature_data = df.drop(["file", "specie"])
    feature_columns = feature_data.columns
    
    # If n_components is None, calculate it as 90% of feature columns
    if n_components is None:
        n_components = max(1, int(0.9 * len(feature_columns)))
    
    # Check available memory
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    
    # Estimate memory requirement for dense matrix
    n_samples = df.shape[0]
    n_features = len(feature_columns)
    estimated_dense_gb = (n_samples * n_features * 8) / (1024 ** 3)  # 8 bytes per float64
    
    print(f"PCA: Available memory: {available_gb:.2f} GB")
    print(f"PCA: Estimated memory need: {estimated_dense_gb:.2f} GB")
    
    # Convert features to numpy array, handling memory constraints
    if estimated_dense_gb > available_gb * 0.7:
        # Use incremental PCA for large datasets
        from sklearn.decomposition import IncrementalPCA
        
        print("Using IncrementalPCA due to memory constraints")
        
        # Initialize incremental PCA
        ipca = IncrementalPCA(n_components=min(n_components, min(n_samples, n_features)))
        
        # Process in batches
        batch_size = 100  # Smaller batch size for large datasets
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = feature_data[start_idx:end_idx]
            
            # Convert batch to float and to numpy array
            batch_np = batch.with_columns(
                [pl.col(c).cast(pl.Float64) for c in batch.columns]
            ).to_numpy()
            
            # Partial fit
            ipca.partial_fit(batch_np)
            
            # Clean up
            del batch
            del batch_np
            gc.collect()
        
        # Transform in batches
        pca_results = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = feature_data[start_idx:end_idx]
            
            # Convert batch to float and to numpy array
            batch_np = batch.with_columns(
                [pl.col(c).cast(pl.Float64) for c in batch.columns]
            ).to_numpy()
            
            # Transform batch
            batch_result = ipca.transform(batch_np)
            pca_results.append(batch_result)
            
            # Clean up
            del batch
            del batch_np
            gc.collect()
        
        # Combine results
        pca_result = np.vstack(pca_results)
        explained_variance_ratio = ipca.explained_variance_ratio_
        
    else:
        # Use regular PCA for smaller datasets
        # Convert features to float and to numpy array
        feature_np = feature_data.with_columns(
            [pl.col(c).cast(pl.Float64) for c in feature_columns]
        ).to_numpy()
        
        # Apply standard PCA
        pca = PCA(n_components=min(n_components, min(feature_np.shape)))
        pca_result = pca.fit_transform(feature_np)
        explained_variance_ratio = pca.explained_variance_ratio_
    
    # Create column names for PCA components
    pca_columns = [f"PC{i+1}" for i in range(pca_result.shape[1])]
    
    # Create DataFrame with PCA results
    vectors_df = pl.DataFrame(pca_result, schema=pca_columns)
    
    # Add back the metadata columns
    vectors_df = vectors_df.with_columns([
        pl.Series("file", file_column.to_numpy()),
        pl.Series("specie", specie_column.to_numpy())
    ])
    vectors_df = vectors_df.select(["file", "specie"] + pca_columns)
    
    # Add metadata
    vectors_df._variance_explained = explained_variance_ratio.sum()
    vectors_df._pca_time = time.time() - start_time
    
    return vectors_df