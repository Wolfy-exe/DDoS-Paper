from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from typing import Iterable, List
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import shuffle as sk_shuffle
from config import config
from csvFunctions import loadCsv, saveDataFrameByLabel, makeRunDirectory, saveConfigToOutputDir
from synthesizeLines import synthesizeLabels

"""
preprocessing.py

Packages used:
- re: regex for sanitizing filenames.
- pandas: tabular data loading/cleaning (CSV, columns, filtering, grouping).
- numpy: numeric ops (Inf/NaN handling, chunk splitting, synthetic interpolation).
- scikit-learn (sklearn.utils.shuffle): reproducible shuffling before chunking.
"""

def collectCsvFiles(path: Path) -> List[Path]:
    """Collects all CSV files from the given path, which can be a single file or a directory. If it's a directory, it recursively searches for CSV files. Returns a sorted list of file paths."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.csv") if p.is_file())
    raise FileNotFoundError(f"Input path not found: {path}")

def stripColumnNames(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading/trailing whitespace from column names to ensure consistent access."""
    cleanedFrame = df.copy()
    cleanedFrame.columns = cleanedFrame.columns.str.strip()
    return cleanedFrame

def replaceInfAndNan(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces Inf and -Inf with NaN, then fills all NaN with 0. This is done to handle any infinite values that may arise during preprocessing, ensuring the resulting DataFrame contains only finite numbers."""
    cleanedFrame = df.copy()
    cleanedFrame = cleanedFrame.replace([np.inf, -np.inf], np.nan)
    return cleanedFrame.fillna(0)

def dropColumnsIfPresent(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drops specified columns from the DataFrame if they exist, ignoring any that are not present."""
    existingColumns = [columnName for columnName in columns if columnName in df.columns]
    frameWithoutDroppedColumns = df.drop(columns=existingColumns, errors="ignore")
    return frameWithoutDroppedColumns.loc[:, ~frameWithoutDroppedColumns.columns.duplicated(keep="first")]

def filterLabel(df: pd.DataFrame, labelValue: str) -> pd.DataFrame:
    """Returns a new DataFrame containing only rows where the 'Label' column matches the specified labelValue (case-insensitive)."""
    return df[df["Label"].astype(str).str.upper() == str(labelValue).upper()].copy()

def fitScaler(preprocessed_dir: Path) -> MaxAbsScaler:
    """Fit MaxAbsScaler on all preprocessed label files without loading everything at once."""
    scaler = MaxAbsScaler()
    csv_files = sorted([f for f in preprocessed_dir.glob("*.csv") if f.is_file()])

    for csv_file in csv_files:
        df = loadCsv(csv_file)
        X = df.iloc[:, :-1].values
        scaler.partial_fit(X)

    return scaler

def applyScalerToPreprocessedFiles(preprocessed_dir: Path, scaler: MaxAbsScaler) -> None:
    """Apply fitted scaler to all preprocessed label files."""
    csv_files = sorted([f for f in preprocessed_dir.glob("*.csv") if f.is_file()])

    for csv_file in csv_files:
        prescaled = loadCsv(csv_file)
        label_col = prescaled["Label"].copy()
        X = prescaled.iloc[:, :-1].values
        X_scaled = scaler.transform(X)

        df_scaled = pd.DataFrame(X_scaled, columns=prescaled.columns[:-1])
        df_scaled["Label"] = label_col.values
        df_scaled.to_csv(csv_file, index=False)

        del prescaled, label_col, X, X_scaled, df_scaled

def preprocessDataFrame(df: pd.DataFrame)-> pd.DataFrame:
    """Preprocesses a raw DataFrame by stripping column names, replacing Inf/NaN, and dropping specified columns."""
    processedFrame = stripColumnNames(df)
    processedFrame = replaceInfAndNan(processedFrame)
    processedFrame = dropColumnsIfPresent(processedFrame, config.preprocessing.columns_to_drop)
    return processedFrame

def getLastPreprocessedRunDir(preprocessedTypesDir: Path) -> Path:
    """Finds the most recent run directory in the preprocessed_types directory."""
    existingPreprocessedDirs = sorted([d for d in preprocessedTypesDir.glob("run_*") if d.is_dir()])
    if not existingPreprocessedDirs:
        raise FileNotFoundError(f"No existing preprocessed run directories found in {preprocessedTypesDir}")
    return existingPreprocessedDirs[-1]

def generateChunks() -> None:
    """Loads each preprocessed label file, splits it into equal chunks based on config."""
    preprocessed_dir = getLastPreprocessedRunDir(Path(config.general.generated_files_base_dir) / "preprocessed_types")
    chunks_dir = Path(config.general.generated_files_base_dir) / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed types directory not found: {preprocessed_dir}")

    if config.preprocessing.keep_preprocessed_chunks:
        existingChunkFiles = sorted([d for d in chunks_dir.glob("run_*") if d.is_dir()], reverse=True)
        if existingChunkFiles:
            latestChunkFile = existingChunkFiles[0]
            print(f"Using existing chunks from {latestChunkFile}...")
            return

    print(f"Generating chunks...")
    run_dir = makeRunDirectory(chunks_dir)
    csv_files = sorted([f for f in preprocessed_dir.glob("*.csv") if f.is_file()])
    chunk_count = config.general.preprocessed_chunks_count

    for csv_file in csv_files:
        print(f"Processing preprocessed label file: {csv_file.name}...")
        df = loadCsv(csv_file)

        if config.general.binary_classification:
            df.loc[df["Label"].astype(str).str.upper() != "BENIGN", "Label"] = "ATTACK"

        shuffled_df = sk_shuffle(df, random_state=config.general.random_seed).reset_index(drop=True)
        split_indices = np.array_split(np.arange(len(shuffled_df)), chunk_count)

        for chunk_num, indices in enumerate(split_indices):
            chunk_df = shuffled_df.iloc[indices].copy()
            chunk_filename = run_dir / f"chunk_{chunk_num}.csv"
            shouldWriteHeader = not chunk_filename.exists()
            chunk_df.to_csv(chunk_filename, index=False, mode="a", header=shouldWriteHeader)

    shuffleChunks()
    saveConfigToOutputDir(run_dir)

def shuffleChunks() -> None:
    """Loads each chunk file, shuffles it, and saves it back to the same location."""
    chunks_dir = Path(config.general.generated_files_base_dir) / "chunks"
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    lastChunks = sorted([d for d in chunks_dir.glob("run_*") if d.is_dir()], reverse=True)
    chunk_files = sorted([f for f in lastChunks[0].glob("*.csv") if f.is_file()])

    for chunk_file in chunk_files:
        df = loadCsv(chunk_file)
        shuffled_df = sk_shuffle(df, random_state=config.general.random_seed)
        shuffled_df.to_csv(chunk_file, index=False)
        del df, shuffled_df

def preprocessData() -> None:
    '''Function that generates preprocessed chunks from raw CSVs, with the ability to generate synthetic lines if enabled in config.'''
    # Basic validation of config settings before starting preprocessing
    if not config.preprocessing.execute_preprocessing:
        print("Preprocessing is disabled in config.")
        return
    if not config.preprocessing.dataset_2019_path:
        raise ValueError("DATASET_2019_PATH in config.py is required.")
    if config.preprocessing.use_2017_benign and not config.preprocessing.dataset_2017_path:
        raise ValueError("DATASET_2017_PATH in config.py is required when USE_2017_BENIGN is set to True.")
    if config.preprocessing.synth_data and not config.preprocessing.synth_labels:
        raise ValueError("SYNTH_LABELS in config.py must contain at least one label when SYNTH_DATA is set to True.")
    if config.preprocessing.undersample_attacks and config.preprocessing.undersample_strategy == "per_type" and config.preprocessing.undersample_samples_per_type <= 0:
        raise ValueError("with UNDERSAMPLE_STRATEGY set to 'per_type', UNDERSAMPLE_SAMPLES_PER_TYPE in config.py must be greater than 0.")

    # Ensure base output directory exists
    baseOutputDir = Path(config.general.generated_files_base_dir)
    baseOutputDir.mkdir(parents=True, exist_ok=True)
    preprocessed_types_dir = baseOutputDir / "preprocessed_types"
    preprocessed_types_dir.mkdir(parents=True, exist_ok=True)

    # Logic to handle using the last preprocessed files or starting fresh based on config settings
    if config.preprocessing.keep_preprocessed_types:
        # idea is that if keep true, then it grabs the latest run (preprocessed_types/run_YYYYMMDD_HHMMSS) and uses those preprocessed types instead of regenerating from raw csvs
        existingPreprocessedDirs = sorted([d for d in preprocessed_types_dir.glob("run_*") if d.is_dir()])
        if existingPreprocessedDirs:
            latestPreprocessedDir = existingPreprocessedDirs[-1]
            print(f"Using existing preprocessed types from {latestPreprocessedDir}...")
            generateChunks()
            return

    print(f"Starting preprocessing...")
    run_dir = makeRunDirectory(preprocessed_types_dir)

    # Preprocess 2019 dataset
    print(f"Processing raw CSV files from {config.preprocessing.dataset_2019_path}...")
    dataset2019Paths = collectCsvFiles(Path(config.preprocessing.dataset_2019_path))
    for dataset2019Path in dataset2019Paths:
        dataset2019Frame = loadCsv(dataset2019Path)
        dataset2019Frame = preprocessDataFrame(dataset2019Frame)
        saveDataFrameByLabel(dataset2019Frame, run_dir)
        del dataset2019Frame

    # Oversampling BENIGN samples from 2017 dataset if enabled in config
    if config.preprocessing.use_2017_benign and (baseOutputDir / "preprocessed_types").exists():
        print(f"Adding BENIGN samples from 2017 dataset at {config.preprocessing.dataset_2017_path}...")
        dataset2017Paths = collectCsvFiles(Path(config.preprocessing.dataset_2017_path))
        for dataset2017Path in dataset2017Paths:
            dataset2017Frame = loadCsv(dataset2017Path)
            dataset2017Frame = preprocessDataFrame(dataset2017Frame)
            dataset2017Frame = filterLabel(dataset2017Frame, labelValue="BENIGN")
            saveDataFrameByLabel(dataset2017Frame, run_dir)
            del dataset2017Frame

    # Oversampling logic based on config settings
    if config.preprocessing.synth_data and config.preprocessing.synth_labels:
        print(f"Generating synthetic samples for labels: {config.preprocessing.synth_labels}")
        synthesizeLabels(run_dir)

    # Undersampling logic based on config settings
    if config.preprocessing.undersample_attacks:
        print(f"Undersampling attack types with strategy '{config.preprocessing.undersample_strategy}'...")
        csv_files = sorted(run_dir.glob("*.csv"))

        # Determine target sample count based on strategy
        if config.preprocessing.undersample_strategy == "per_type":
            targetSampleCount = config.preprocessing.undersample_samples_per_type
        else:  # "benign_as_limit"
            # Find BENIGN count and divide by number of attack types
            benign_count = 0
            attack_type_count = 0

            for csv_file in csv_files:
                if "BENIGN" in csv_file.stem.upper():
                    benign_df = loadCsv(csv_file)
                    benign_count = len(benign_df)
                else:
                    attack_type_count += 1

            targetSampleCount = benign_count // max(attack_type_count, 1) if attack_type_count > 0 else benign_count

        for csv_file in csv_files:
            labelFrame = loadCsv(csv_file)
            desiredCount = targetSampleCount

            if config.general.binary_classification and config.preprocessing.multiply_benign and "BENIGN" in csv_file.stem.upper():
                desiredCount = targetSampleCount * max(len(csv_files) - 1, 0)

            if len(labelFrame) > desiredCount:
                labelFrame = labelFrame.sample(
                    n=desiredCount,
                    replace=False,
                    random_state=config.general.random_seed,
                )

            labelFrame = sk_shuffle(labelFrame, random_state=config.general.random_seed).reset_index(drop=True)
            labelFrame.to_csv(csv_file, index=False)
            del labelFrame

    # Fit and apply MaxAbsScaler BEFORE chunking
    print("Fitting MaxAbsScaler on all preprocessed data...")
    scaler = fitScaler(run_dir)
    print("Applying scaler to preprocessed files...")
    applyScalerToPreprocessedFiles(run_dir, scaler)
    del scaler

    saveConfigToOutputDir(run_dir)
    generateChunks()

if __name__ == "__main__":
    preprocessData()