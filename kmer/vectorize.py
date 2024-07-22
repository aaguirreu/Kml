from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def calculate_tfidf(kmer_lists, k):
    tfidf_vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, ngram_range=(k, k))
    tfidf_matrix = tfidf_vectorizer.fit_transform(kmer_lists)
    return tfidf_matrix

def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true, y_pred, y_proba):
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
    return f1, mcc, auc

def vectorization(tsv_file):
    df = pd.read_csv(tsv_file, delimiter='\t')

    # Preserve the original labels
    original_labels = df['label']

    # One-hot encode the 'label' and 'ID_genome' columns
    encoder = OneHotEncoder()
    encoded_labels = encoder.fit_transform(df[['label', 'ID_genome']]).toarray()

    # Add encoded columns to the dataframe
    encoded_columns = encoder.get_feature_names_out(['label', 'ID_genome'])
    encoded_df = pd.DataFrame(encoded_labels, columns=encoded_columns, index=df.index)

    # Combine with the original dataframe, dropping the original 'label' and 'ID_genome' columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    df = pd.concat([encoded_df, df[numeric_columns]], axis=1)

    # Check if we have enough samples to split
    if len(df) > 1 and all(original_labels.value_counts() > 1):
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=original_labels)
    else:
        print("Not enough samples to split. Using entire dataset for both training and testing.")
        train_df = test_df = df

    # Separate features and labels
    X_train = train_df.drop(columns=encoded_columns)
    y_train = train_df[encoded_columns].values.argmax(axis=1)
    X_test = test_df.drop(columns=encoded_columns)
    y_test = test_df[encoded_columns].values.argmax(axis=1)

    # Normalize data
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    
    model = RandomForestClassifier()
    trained_model = train_model(X_train, y_train, model)
    y_pred = trained_model.predict(X_test)
    y_proba = trained_model.predict_proba(X_test)
    f1, mcc, auc = evaluate_model(y_test, y_pred, y_proba)
    
    print(f"F1 Score: {f1}, MCC: {mcc}, AUC: {auc}")