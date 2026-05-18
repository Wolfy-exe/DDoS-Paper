# Synthesize Lines for Data Augmentation
from pathlib import Path
import numpy as np
import pandas as pd
from config import config
from csvFunctions import getFilePathForPreprocessedLabel, saveDataFrameByLabel, loadCsv

def synthesizeLabelInterpolation(
    labelFrame: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates additional synthetic rows for one label via interpolation between
    random row pairs on numeric columns. Non-numeric columns are copied.
    """
    currentCount = len(labelFrame)
    if currentCount >= config.preprocessing.synth_target or currentCount == 0:
        return labelFrame

    randomGenerator = np.random.default_rng(config.general.random_seed)
    additionalRowsNeeded = config.preprocessing.synth_target - currentCount

    numericColumns = labelFrame.select_dtypes(include=[np.number]).columns.tolist()
    numericColumns = [columnName for columnName in numericColumns if columnName != "Label"]
    nonNumericColumns = [
        columnName for columnName in labelFrame.columns if columnName not in numericColumns and columnName != "Label"
    ]

    if not numericColumns:
        duplicatedRows = labelFrame.sample(n=additionalRowsNeeded, replace=True, random_state=config.general.random_seed).copy()
        return pd.concat([labelFrame, duplicatedRows], ignore_index=True)

    sourceIndexesA = randomGenerator.integers(0, currentCount, size=additionalRowsNeeded)
    sourceIndexesB = randomGenerator.integers(0, currentCount, size=additionalRowsNeeded)
    interpolationWeights = randomGenerator.random(size=additionalRowsNeeded).reshape(-1, 1)

    numericSourceA = labelFrame.iloc[sourceIndexesA][numericColumns].to_numpy(dtype=float)
    numericSourceB = labelFrame.iloc[sourceIndexesB][numericColumns].to_numpy(dtype=float)
    syntheticNumericValues = interpolationWeights * numericSourceA + (1.0 - interpolationWeights) * numericSourceB

    baseRows = labelFrame.sample(n=additionalRowsNeeded, replace=True, random_state=config.general.random_seed).reset_index(drop=True)
    syntheticFrame = baseRows.copy()
    syntheticFrame[numericColumns] = syntheticNumericValues
    for columnName in nonNumericColumns:
        syntheticFrame[columnName] = baseRows[columnName]
    syntheticFrame["Label"] = labelFrame["Label"].iloc[0]

    return pd.concat([labelFrame, syntheticFrame], ignore_index=True)

def synthesizeLabels(basePath: Path) -> None:
    if not config.preprocessing.synth_data:
        return
    if not config.preprocessing.synth_labels:
        return
    if config.preprocessing.synth_target <= 0:
        return

    for labelName in config.preprocessing.synth_labels:
        filePath = getFilePathForPreprocessedLabel(labelName, basePath)
        labelFrame = loadCsv(filePath)

        synthesizedFrame = synthesizeLabelInterpolation(labelFrame)

        saveDataFrameByLabel(synthesizedFrame, basePath)
