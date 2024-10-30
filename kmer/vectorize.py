from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from kmer import log_step
import pandas as pd
import numpy as np

def extract_taxonomy(label):
    # Separar la cadena en partes
    if '[' in label:
        label = label.split('[')[0]  # Mantener solo la parte antes del '['
    if ';' in label:
        parts = label.split(';')
    elif ',' in label:
        label = label.split(',')[0]
        parts = label.split(' ')
    # Crear un diccionario para almacenar los niveles taxonómicos
    taxonomy = {}
    if '__' in label:
        for part in parts:
            # Separar el nivel del valor
            level, value = part.split('__') if '__' in part else (part, "")
            taxonomy[level] = value
    else:
        taxonomy['s'] = ' '.join(parts[:2])
    return taxonomy

def vectorize_kmers(kmer_counts_df):
    vector_list = []

    for index, row in kmer_counts_df.iterrows():
        total_kmers = row[2:].sum() 
        if total_kmers > 0:
            normalized = row[2:] / total_kmers 
        else:
            normalized = row[2:] 
        
        vector_list.append(normalized)

    vector_df = pd.DataFrame(vector_list, columns=kmer_counts_df.columns[2:])
    
    vector_df.insert(0, 'ID_genome', kmer_counts_df['ID_genome'])
    vector_df.insert(1, 'label', kmer_counts_df['label'])
    
    return vector_df

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

def classification(vector_df):
    try:
        taxonomy_df = vector_df['label'].apply(extract_taxonomy).apply(pd.Series)
        taxonomy_df
        vector_df['label'] = taxonomy_df['s']
        vector_df = vector_df.rename(columns={'label': 'specie'})
        del taxonomy_df
        
        single_value_classes = vector_df['specie'].value_counts()[vector_df['specie'].value_counts() < 2].index.tolist()    
        for clase in single_value_classes:
            row_to_duplicate = vector_df[vector_df['specie'] == clase]
            if not row_to_duplicate.empty:
                vector_df = pd.concat([vector_df, row_to_duplicate], ignore_index=True)
        vector_df

        num_classes = vector_df['specie'].nunique()
        total_samples = len(vector_df)

        test_size = max(0.2, num_classes / total_samples)
        log_step(f"Test size a utilizar: {test_size}")

        encoder = LabelEncoder()
        vector_df['specie'] = encoder.fit_transform(vector_df['specie'])
        vector_df = vector_df.drop(columns=['ID_genome'])

        train_df, test_df = train_test_split(vector_df, test_size=test_size, stratify=vector_df['specie'])

        # Separar características y etiquetas
        X_train = train_df.drop(columns=['specie'])
        y_train = train_df['specie']
        X_test = test_df.drop(columns=['specie'])
        y_test = test_df['specie']

        # Normalize data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)

        model = RandomForestClassifier()
        trained_model = model.fit(X_train, y_train)

        y_pred = trained_model.predict(X_test)
        y_proba = trained_model.predict_proba(X_test)

        f1 = f1_score(y_test, y_pred, average='weighted')
        log_step(f"F1 Score: {f1}")
        mcc = matthews_corrcoef(y_test, y_pred)
        log_step(f"MCC: {mcc}")
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        log_step(f"AUC: {auc}")
    except ValueError as e:
        log_step(f"Error: {e}")
        auc_score = None