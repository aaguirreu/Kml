import hashlib
import polars as pl
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder

def time_vectorization(func):
    """Decorator to measure execution time of vectorization functions"""
    def wrapper(df, *args, **kwargs):
        start_time = time.time()
        result = func(df, *args, **kwargs)
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

# Dictionary containing the vectorization methods to evaluate
vectorization_methods = {
    'K-mer Frequency': vectorize_kmer_frequency,
    'TF-IDF': vectorize_tfidf,
    'Relative Frequency': vectorize_relative_frequency,
    # 'K-mer Hashing': vectorize_kmer_hashing,
    # 'One-hot': vectorize_onehot,
}