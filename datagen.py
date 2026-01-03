import string
from sklearn.datasets import make_classification, make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import random as sparse_random
import numpy as np
import pandas as pd
import kagglehub
import os
import shutil
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def generate_classification_data(
    n_samples=100_000,
    n_features=50,
    n_informative=10,
    n_classes=2,
    sparsity=0.0,
    random_state=42,
):
    """
    Generate synthetic classification data.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    n_informative : int
        Number of informative features.
    n_classes : int
        Number of classes.
    sparsity : float
        Fraction of features set to zero (0 = dense, 0.9 = 90% zeros).
    random_state : int
        Random seed.

    Returns
    -------
    (X, y) : tuple of np.ndarray
        Feature matrix and target vector.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=random_state,
    )

    if sparsity > 0:
        mask = np.random.rand(*X.shape) > sparsity
        X = X * mask

    return X, y

def generate_clustering_data(
    n_samples=100_000,
    n_features=50,
    centers=10,
    cluster_std=1.0,
    sparsity=0.0,
    random_state=42,
    
):
    """
    Generate synthetic clustering data.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    centers : int
        Number of cluster centers.
    cluster_std : float
        Standard deviation of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    (X, y) : tuple of np.ndarray
        Feature matrix and cluster labels.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
        return_centers=False,
    )
    if sparsity > 0:
        mask = np.random.rand(*X.shape) > sparsity
        X = X * mask
        
    return X, y

def loadFromParquet(file_path: str = f".{os.sep}data{os.sep}",files : tuple[str,str]=("classifier_results.parquet","clustering_results.parquet")) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Load results from parquet files.

    Parameters
    ----------
    file_path : str
        Directory path where parquet files are stored.
    files : tuple of str
        Filenames of the parquet files to load.

    Returns
    -------
    dataframes : tuple of pd.DataFrame
        Loaded dataframes for each file.
    """
    dataframes = []
    for file in files:
        full_path = os.path.join(file_path, file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        df = pd.read_parquet(full_path)
        dataframes.append(df)
        
    df_classifier,df_clustering = dataframes
     
    return df_classifier.reset_index(drop=True), df_clustering.reset_index(drop=True)

def get_dataset():
    df = pd.read_csv(f".{os.sep}Datasets{os.sep}spam_ham_dataset.csv")
    df = _pre_process_dataset(df)
    return df

def _pre_process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    import nltk
    import os
    
    # Resolve path RELATIVE to THIS file (datagen.py)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")

    # Ensure the folder exists
    os.makedirs(NLTK_DIR, exist_ok=True)

    # Make NLTK use this folder
    nltk.data.path.insert(0, NLTK_DIR)


    # Download resources into this relative folder
    nltk.download("punkt", download_dir=NLTK_DIR)
    nltk.download("stopwords", download_dir=NLTK_DIR)
    
    print("NLTK DIR:", NLTK_DIR)
    print("NLTK PATH:", nltk.data.path)

    print("RESULT punkt:", nltk.download("punkt", download_dir=NLTK_DIR))
    print("RESULT stopwords:", nltk.download("stopwords", download_dir=NLTK_DIR))


    #lowercasing
    df["text"] = df["text"].apply(lambda text : text.lower())
    #tokenizing
    df["tokens"]=df["text"].apply(lambda text : word_tokenize(text))
    #punctuation removal
    df["tokens"]=df["tokens"].apply(lambda tokens :[word for word in tokens if word not in string.punctuation])
    #stop word removal
    stop_words=set(stopwords.words('english'))
    df["tokens"]=df["tokens"].apply(lambda tokens : [word for word in tokens if word not in stop_words])

    #Converting from tokens back to strings
    df["text_processed"]=df["tokens"].apply(lambda tokens : " ".join(tokens))
    
        #initialisation of vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    #fitting and transforming the 'text_processed' column
    tfidf_features = tfidf_vectorizer.fit_transform(df['text_processed'])

    #Converting the TF-IDF features to a dataframe
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    return tfidf_df