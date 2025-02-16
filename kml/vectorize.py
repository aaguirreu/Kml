import hashlib
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder

def vectorize_kmer_frequency(df):
    """
    In this method no transformation is applied,
    assuming the DataFrame already contains the absolute k-mer counts.
    """
    return df

def vectorize_tfidf(df: pl.DataFrame) -> pl.DataFrame:
    """
    Realiza la transformación TF-IDF sobre las columnas de k-mers.
    
    TF (Term Frequency): cada valor se divide por la suma de la fila.
    IDF (Inverse Document Frequency): para cada k-mer se calcula 
        log((n_rows + 1) / (doc_freq + 1)) + 1,
    donde doc_freq es la cantidad de filas en las que el k-mer es mayor a 0.
    
    Parámetros:
        df (pl.DataFrame): DataFrame con columnas "file", "specie" y k-mer counts.
        
    Retorna:
        pl.DataFrame: DataFrame con las columnas "file", "specie" y los valores TF-IDF para cada k-mer.
    """
    # Extraer las columnas "specie" y "file"
    specie_column = df["specie"]
    file_column = df["file"]
    
    # Seleccionar las columnas de k-mers
    kmer_data = df.drop(["file", "specie"])
    
    # Convertir los k-mers a float y extraer como array de NumPy
    kmer_np = kmer_data.with_columns([pl.col(c).cast(pl.Float64) for c in kmer_data.columns]).to_numpy()
    
    # Crear y aplicar el transformer TF-IDF
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(kmer_np)
    
    # Convertir la matriz TF-IDF a array denso
    dense_tfidf = tfidf_matrix.toarray()
    
    # Crear DataFrame de Polars con los nombres de columnas originales de los k-mers
    vectors_df = pl.DataFrame(dense_tfidf, schema=kmer_data.columns)
    
    # Insertar las columnas "file" y "specie" al final y luego reordenarlas
    vectors_df = vectors_df.with_columns([
        pl.Series("file", file_column.to_numpy()),
        pl.Series("specie", specie_column.to_numpy())
    ])
    vectors_df = vectors_df.select(["file", "specie"] + kmer_data.columns)
    
    return vectors_df

def vectorize_relative_frequency(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula la frecuencia relativa de cada k-mer en cada muestra.

    Para cada fila se suma el total de k-mers y se divide cada valor por dicha suma.

    Parámetros:
        df (pl.DataFrame): DataFrame con columnas "file", "specie" y k-mer counts.

    Retorna:
        pl.DataFrame: DataFrame con las columnas "file", "specie" y los k-mers convertidos
                      a frecuencias relativas.
    """
    id_cols = ["file", "specie"]
    numeric_cols = [col for col in df.columns if col not in id_cols]

    # Calcular la suma horizontal de los k-mers en cada fila.
    df = df.with_columns(pl.sum_horizontal(numeric_cols).alias("row_sum"))

    # Dividir todas las columnas numéricas por 'row_sum' en una sola operación.
    df = df.with_columns(
        [(pl.col(col) / pl.col("row_sum")).alias(col) for col in numeric_cols]
    ).drop("row_sum")

    return df


def vectorize_kmer_hashing(df: pl.DataFrame, num_buckets: int = 10) -> pl.DataFrame:
    """
    Realiza la vectorización mediante hashing de k-mers.
    
    Para cada columna de k-mer se calcula un bucket (usando SHA256) y se agregan los
    conteos de todos los k-mers que caen en el mismo bucket.
    
    Esta implementación itera fila a fila (lo que puede ser menos eficiente en datasets muy grandes).
    
    Parámetros:
        df (pl.DataFrame): DataFrame con columnas "file", "specie" y k-mer counts.
        num_buckets (int): Número de buckets a utilizar.
    
    Retorna:
        pl.DataFrame: DataFrame con columnas "file", "specie" y buckets (e.g., bucket_0, bucket_1, ...).
    """
    id_cols = ["file", "specie"]
    numeric_cols = [col for col in df.columns if col not in id_cols]
    
    # Precalcular la asignación de cada k-mer a un bucket
    bucket_map = {col: int(hashlib.sha256(col.encode()).hexdigest(), 16) % num_buckets for col in numeric_cols}
    
    # Función que se aplicará a cada fila para agrupar los conteos por bucket
    def hash_row(row: dict) -> list:
        buckets = [0] * num_buckets
        for col in numeric_cols:
            buckets[bucket_map[col]] += row[col]
        return buckets

    # Aplicar la función de hashing fila a fila; el resultado es una columna de listas
    df = df.with_columns(
        pl.struct(numeric_cols)
        .apply(lambda s: hash_row(s), return_dtype=pl.List(pl.Int64))
        .alias("hashed")
    )
    
    # Expandir la columna de listas en columnas separadas para cada bucket
    for i in range(num_buckets):
        df = df.with_columns(pl.col("hashed").arr.get(i).alias(f"bucket_{i}"))
    
    df = df.drop("hashed")
    return df

def vectorize_onehot(df: pl.DataFrame) -> pl.DataFrame:
    """
    Realiza la codificación one-hot en las columnas de k-mers.
    
    Convierte los conteos a valores binarios (1 si el k-mer está presente, 0 si no).
    
    Parámetros:
        df (pl.DataFrame): DataFrame con columnas "file", "specie" y k-mer counts.
        
    Retorna:
        pl.DataFrame: DataFrame con las columnas "file", "specie" y los k-mers codificados en binario.
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