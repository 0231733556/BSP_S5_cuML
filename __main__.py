from BSP_S5_cuML.models import ClassifierWrapper, ClusteringWrapper, train_model,set_defaults,set_log_level, use_accel # type: ignore
from BSP_S5_cuML.datagen import generate_classification_data, generate_clustering_data # type: ignore
from sklearn.model_selection import train_test_split
import logging as LOG
import pandas as pd
import os

classifiers = ["svc", "random_forest","logistic"]
clusterers = ["kmeans", "dbscan","agglomerative"]

SEED = 42
CLASSIFIER= classifiers[2]
CLUSTER= clusterers[2]


def write_results_to_parquet(results: dict, filename: str) -> None:
    df = pd.DataFrame([results])
    if os.path.exists(filename):
        df_existing = pd.read_parquet(filename)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_parquet(filename, index=False)

def record_trial(n_samples:int, n_features:int, sparsity:float, accelerator:str, algorithm:str, results:dict) -> dict:
    row = {
        "n_samples": n_samples,
        "n_features": n_features,
        "sparsity": sparsity,
        "accelerator": accelerator,
        "median_time" : results.get("median", None),
        "mean_time" : results.get("mean", None),
        "algorithm" : algorithm,
    }
    return row

def run_project(n_samples=100_000, n_features=20, n_informative=15, n_classes=2,sparsity=0.0,centers=10,cluster_std=1.0,test_size=0.2, random_state=42):
    set_log_level(LOG.DEBUG)
    classifierXY = generate_classification_data(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_classes=n_classes,sparsity=sparsity, random_state=random_state)
    classifierSplit = train_test_split(classifierXY[0], classifierXY[1], test_size=test_size, random_state=random_state)
    clusteringXY = generate_clustering_data(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)

    setting={
        "classifier": {
            "estimator_name": CLASSIFIER,
            "use_scaler": True,
            "random_state": SEED,
            "n_estimators": 100,
            "max_iter": 1000,
            "probability": True,
        },
        "clustering": {
            "algorithm_name": CLUSTER,
            "n_clusters": 3,
            "use_scaler": True,
            "random_state": SEED,
            "algorithm_params": None,
        },
    }
    set_defaults(setting)

    rfc = ClassifierWrapper()
    kmc = ClusteringWrapper()
    resultClassifier=train_model(model=rfc,X=classifierSplit[0],y=classifierSplit[2],X_val=classifierSplit[1],y_val=classifierSplit[3],timing=True,trials=5)
    resultClustering=train_model(kmc,X=clusteringXY[0],timing=True,trials=5)
    row_classifier=record_trial(n_samples=n_samples, n_features=n_features, sparsity=sparsity, accelerator="GPU1", results=resultClassifier, algorithm=CLASSIFIER)
    row_clustering=record_trial(n_samples=n_samples, n_features=n_features, sparsity=sparsity, accelerator="GPU1", results=resultClustering, algorithm=CLUSTER)
    write_results_to_parquet(row_classifier, "." + os.sep + "BSP_S5_cuML" + os.sep + "data" +os.sep+"classifier_results.parquet")
    write_results_to_parquet(row_clustering, "." + os.sep + "BSP_S5_cuML" + os.sep + "data" +os.sep+"clustering_results.parquet")
    print("Classifier results:", resultClassifier)
    print("Clustering results:", resultClustering)
    
    
    
def main () :
    number_samples = [1000,10_000]
    n_features = [64,128,256,512,1024,2048,4096]
    sparsities = [0.0,0.25,0.5,0.75,0.9]
    
    for n_samples in number_samples:
        for n_feature in n_features:
            for sparsity in sparsities:
                run_project(n_samples=n_samples, n_features=n_feature, n_informative=15, n_classes=2,sparsity=sparsity,centers=10,cluster_std=1.0,test_size=0.2, random_state=42)



if __name__ == "__main__":
    main()