import json, numpy as np, pandas as pd
from config import config
from pathlib import Path
from typing import Dict, Any
from sklearn.linear_model import SGDClassifier
from csvFunctions import evaluateMetrics, loadCsv, makeRunDirectory

def loadBestHyperparameters(outputDir: Path) -> Dict[str, Any]:
    hyperparamPath = outputDir / "best_hyperparameters.json"
    if not hyperparamPath.exists():
        raise FileNotFoundError(f"Best hyperparameters file not found: {hyperparamPath}")
    with open(hyperparamPath, "r") as f:
        return json.load(f)

def separateFeaturesAndLabels(chunkData: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = chunkData.iloc[:, :-1].values
    y = chunkData.iloc[:, -1].values
    return X, y

def getChunk(chunkIndex: int) -> pd.DataFrame:
    path = Path(config.general.generated_files_base_dir) / "chunks"
    if not path.exists():
        raise FileNotFoundError(f"Chunk files not found: {path}")
    latestChunkGroup = sorted([d for d in path.glob("run_*") if d.is_dir()], reverse=True)
    if not latestChunkGroup:
        raise FileNotFoundError(f"No chunk run directories found in: {path}")
    chunkPath = latestChunkGroup[0] / f"chunk_{chunkIndex}.csv"
    return loadCsv(chunkPath)

def trainModelOnChunk(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    if hasattr(model, "partial_fit"):
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    else:
        model.fit(X_train, y_train)
    return model

def validateModel() -> None:
    resultsDir = Path(config.general.generated_files_base_dir) / "training_results"
    if not resultsDir.exists():
        raise FileNotFoundError(f"Training results directory not found: {resultsDir}")
    lastRunResults = sorted([d for d in resultsDir.glob("run_*") if d.is_dir()], reverse=True)
    if not lastRunResults:
        raise FileNotFoundError(f"No training run directories found in: {resultsDir}")
    lastRunDir = lastRunResults[0]

    print("Loading best hyperparameters...")
    bestHyperparameters = loadBestHyperparameters(lastRunDir)

    print("Retraining model with best hyperparameters...")
    model = SGDClassifier(
        loss=bestHyperparameters.get("loss", "log_loss"),
        alpha=bestHyperparameters.get("alpha", 0.0001),
        random_state=config.general.random_seed,
        warm_start=False,
    )

    chunkCount = config.general.preprocessed_chunks_count
    trainingChunkIndices = list(range(chunkCount - 2))

    for chunkIdx in trainingChunkIndices:
        trainChunk = getChunk(chunkIdx)
        X_train, y_train = separateFeaturesAndLabels(trainChunk)
        model = trainModelOnChunk(model, X_train, y_train)
        del trainChunk

    print("Validating on final chunk...")
    validationChunk = getChunk(chunkCount - 1)
    X_val, y_val = separateFeaturesAndLabels(validationChunk)
    y_pred = model.predict(X_val)

    validationMetrics = evaluateMetrics(y_val, y_pred)

    print("\nValidation Results:")
    for metricName, metricValue in validationMetrics.items():
        if metricName == "confusion_matrix":
            print(f"{metricName}:")
            for line in metricValue:
                print(f"  {line}")
        else:
            print(f"{metricName}: {metricValue}")

    valOutputDir = Path(config.general.generated_files_base_dir) / "validation_results"
    valOutputDir = makeRunDirectory(valOutputDir)
    valOutputDir.mkdir(parents=True, exist_ok=True)
    with open(valOutputDir / "validation_metrics.json", "w") as f:
        json.dump(validationMetrics, f, indent=2)
    print(f"\nValidation results saved to {valOutputDir}")

if __name__ == "__main__":
    validateModel()