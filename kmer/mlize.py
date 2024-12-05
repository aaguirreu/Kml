from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from kmer import log_step
import pandas as pd
import numpy as np

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    # 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    # 'Supporkt Vector Machine (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
}

def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, auc=True):
    """
    Train and evaluate a model, returning metrics.

    Parameters:
        model: The machine learning model to evaluate.
        X_train, X_test: Feature sets for training and testing.
        y_train, y_test: Labels for training and testing.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # AUC (if probabilities available)
    if auc:
        if y_pred_proba is not None:
            if y_pred_proba.shape[1] == 2:  # Binary classification
                y_pred_proba = y_pred_proba[:, 1]  # Use probabilities for the positive class
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr' if len(set(y_test)) > 2 else None)
        else:
            auc = None

    return {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'AUC': auc,
        'MCC': mcc
    }

def run_models(df):
    # Data Preparation
    X = df.iloc[:, 2:]  # Assuming k-mer counts start from the 3rd column
    y = df['specie']

    # Normalize k-mer counts
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Evaluate each model
    all_results = {}
    for model_name, model in models.items():
        log_step(f"Evaluating {model_name}...")
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        all_results[model_name] = results  # Almacenar resultados
        log_step(f"Results for {model_name}:")
        for metric_name, value in results.items():
            if value is not None:
                log_step(f"  {metric_name}: {value:.4f}")
        log_step("-" * 40)

    return X_train, X_test, y_train, y_test, all_results

def classification(vector_df):
    try:
        taxonomy_df = vector_df['label'].apply(extract_taxonomy).apply(pd.Series)
        vector_df['label'] = taxonomy_df['s']
        vector_df = vector_df.rename(columns={'label': 'specie'})
        del taxonomy_df
        
        single_value_classes = vector_df['specie'].value_counts()[vector_df['specie'].value_counts() < 2].index.tolist()    
        for clase in single_value_classes:
            row_to_duplicate = vector_df[vector_df['specie'] == clase]
            if not row_to_duplicate.empty:
                vector_df = pd.concat([vector_df, row_to_duplicate], ignore_index=True)
        #vector_df

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