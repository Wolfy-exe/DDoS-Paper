import json, itertools, numpy as np, pandas as pd
from config import config
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.linear_model import SGDClassifier
from csvFunctions import saveConfigToOutputDir, loadCsv, evaluateMetrics

def separateFeaturesAndLabels(
    chunkData: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    X = chunkData.iloc[:, :-1].values
    y = chunkData.iloc[:, -1].values
    return X, y

def getChunk(chunkIndex: int) -> pd.DataFrame:
    chunksPath = Path(config.general.generated_files_base_dir) / "chunks"
    latestChunkGroup = sorted([d for d in chunksPath.glob("run_*") if d.is_dir()], reverse=True)
    if latestChunkGroup:
        chunksPath = latestChunkGroup[0] / f"chunk_{chunkIndex}.csv"
    if not chunksPath.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunksPath}")
    return loadCsv(chunksPath)

def trainModelOnChunk(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    if hasattr(model, "partial_fit"):
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    else:
        model.fit(X_train, y_train)
    return model

def isMetricBetter(
    newMetrics: Dict[str, float],
    bestMetrics: Dict[str, float],
    primaryMetric: str,
) -> bool:
    if not bestMetrics:
        return True
    return newMetrics[primaryMetric] > bestMetrics[primaryMetric]

def saveResultsMap(
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    outputFile: Path,
) -> None:
    """
    Save a mapping file in outputDir named results_map.json that maps a
    representation of the hyperparameters to the corresponding metrics.
    If the file exists it will be loaded and updated; otherwise created.
    """
    outputFile.parent.mkdir(parents=True, exist_ok=True)

    # Key: stable JSON string of hyperparameters, Value: metrics dict
    key = json.dumps(hyperparameters, sort_keys=True)

    if outputFile.exists():
        try:
            with open(outputFile, "r") as f:
                results_map = json.load(f)
        except Exception:
            results_map = {}
    else:
        results_map = {}

    results_map[key] = metrics

    with open(outputFile, "w") as f:
        json.dump(results_map, f, indent=2)

def saveResults(
    bestHyperparameters: Dict[str, Any],
    bestMetrics: Dict[str, float],
    outputDir: Path,
) -> None:
    outputDir.mkdir(parents=True, exist_ok=True)

    hyperparamPath = outputDir / "best_hyperparameters.json"
    with open(hyperparamPath, "w") as f:
        json.dump(bestHyperparameters, f, indent=2)

    metricsPath = outputDir / "best_metrics.json"
    with open(metricsPath, "w") as f:
        json.dump(bestMetrics, f, indent=2)

    saveConfigToOutputDir(outputDir)

def generateHyperparameterSets() -> list[Dict[str, Any]]:
    ranges = config.training.hyperparameter_ranges
    keys = ranges.keys()
    values = ranges.values()
    combinations = itertools.product(*values)
    return [dict(zip(keys, combo)) for combo in combinations]

def trainModels() -> None:
    if not (Path(config.general.generated_files_base_dir) / "chunks").exists():
        raise FileNotFoundError("Chunks directory not found")

    print("Starting training process...")
    chunkCount = config.general.preprocessed_chunks_count
    trainingChunkIndices = list(range(chunkCount - 2))
    testingChunkIndex = chunkCount - 2

    outputDir = Path(config.general.generated_files_base_dir) / "training_results" / f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    testChunk = getChunk(testingChunkIndex)
    X_test, y_test = separateFeaturesAndLabels(testChunk)

    hyperparamSets = generateHyperparameterSets()

    bestHyperparameters = None
    bestMetrics = {}

    # Iterate through all hyperparameter combinations
    for hyperparamSet in hyperparamSets:
        print(f"Testing hyperparameters: {hyperparamSet}")

        # Create SGDClassifier with current hyperparameters
        model = SGDClassifier(
            loss=hyperparamSet.get("loss", "log_loss"),
            alpha=hyperparamSet.get("alpha", 0.0001),
            random_state=config.general.random_seed,
            warm_start=False
        )

        # Train on all training chunks using partial_fit
        for chunkIdx in trainingChunkIndices:
            trainChunk = getChunk(chunkIdx)
            X_train, y_train = separateFeaturesAndLabels(trainChunk)
            model = trainModelOnChunk(model, X_train, y_train)
            del trainChunk  # Unload chunk from memory

        # Evaluate on testing chunk
        y_pred = model.predict(X_test)
        currentMetrics = evaluateMetrics(y_test, y_pred)

        
        # Save current hyperparameters and metrics for this run for later analysis
        saveResultsMap(hyperparamSet, currentMetrics, outputDir / f"results_map.json")

        # Check if this is the best so far
        if isMetricBetter(
            currentMetrics,
            bestMetrics,
            config.training.primary_metric,
        ):
            bestMetrics = currentMetrics
            bestHyperparameters = hyperparamSet
            print(f"New best {config.training.primary_metric}: {bestMetrics[config.training.primary_metric]}")

    # Save results
    saveResults(bestHyperparameters, bestMetrics, outputDir)
    print(f"Training complete. Results saved to {outputDir}")

if __name__ == "__main__":
    trainModels()