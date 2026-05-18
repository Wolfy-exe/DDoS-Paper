import json, re, numpy as np, pandas as pd, math
from pathlib import Path
from typing import Dict
from config import config
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def sanitizeFilename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")

def loadCsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"SimillarHTTP": "str"})

def splitByLabel(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    labelGroups: Dict[str, pd.DataFrame] = {}
    for labelValue, labelFrame in df.groupby("Label", dropna=False):
        labelGroups[str(labelValue)] = labelFrame.copy()
    return labelGroups

def getFilePathForPreprocessedLabel(labelName: str, basePath: Path) -> Path:
    sanitizedLabel = sanitizeFilename(labelName)
    for file in basePath.glob(f"preprocessed_{sanitizedLabel}.csv"):
        return file
    raise FileNotFoundError(f"No file found for label '{labelName}' in directory '{basePath}'.")

def saveDataFrameByLabel(df: pd.DataFrame, outputPath: Path) -> None:
    outputPath.mkdir(parents=True, exist_ok=True)

    splitByLabelFrame = splitByLabel(df)
    for labelName, labelFrame in splitByLabelFrame.items():
        sanitizedLabel = sanitizeFilename(labelName)
        labelOutputPath = outputPath / f"preprocessed_{sanitizedLabel}.csv"
        shouldWriteHeader = not labelOutputPath.exists()
        labelFrame.to_csv(labelOutputPath, mode="a", header=shouldWriteHeader, index=False)

def saveConfigToOutputDir(outputDir: Path) -> None:
    configPath = outputDir / "config.json"
    with open(configPath, "w") as f:
        config_dict = {key: vars(value) if hasattr(value, '__dict__') else value 
                       for key, value in vars(config).items()}
        json.dump(config_dict, f, indent=2)

def makeRunDirectory(baseDir: Path) -> Path:
    runDir = baseDir / f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    runDir.mkdir(parents=True, exist_ok=True)
    return runDir

def evaluateMetrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    labels = np.unique(np.concatenate((y_true.astype(str), y_pred.astype(str))))
    if config.general.binary_classification:
        # Convert string labels to numeric for confusion matrix
        label_mapping = {label: idx for idx, label in enumerate(sorted(labels))}
        y_true_numeric = np.array([label_mapping[label] for label in y_true])
        y_pred_numeric = np.array([label_mapping[label] for label in y_pred])
        
        tn, fp, fn, tp = confusion_matrix(y_true_numeric, y_pred_numeric).ravel().tolist()
        
        # Calculate MCC safely
        mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(mcc_denominator) if mcc_denominator > 0 else 0.0
        
        metrics = {
            "confusion_matrix": [
                "true negative: "+str(tn), 
                "false positive: "+str(fp), 
                "false negative: "+str(fn), 
                "true positive: "+str(tp)
            ],
            "accuracy": accuracy_score(y_true_numeric, y_pred_numeric),
            "recall": recall_score(y_true_numeric, y_pred_numeric, zero_division=0),
            "precision": precision_score(y_true_numeric, y_pred_numeric, zero_division=0),
            "specificity": (tn) / (tn + fp) if (tn + fp) > 0 else 0,
            "NPV": (tn) / (tn + fn) if (tn + fn) > 0 else 0,
            "f1": f1_score(y_true_numeric, y_pred_numeric, zero_division=0),
            "ROC AUC": roc_auc_score(y_true_numeric, y_pred_numeric),
            "MCC": mcc,
        }
        return metrics
    else:
        return {
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
        }