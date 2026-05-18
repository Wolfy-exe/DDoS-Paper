from pydantic import BaseModel
from typing import List

class GeneralSettings(BaseModel):
    """General settings for the application."""
    random_seed: int = 42 # Used for reproducibility across all random operations
    binary_classification: bool = True # Whether to treat all attack types as a single "Attack" class vs keeping them separate
    generated_files_base_dir: str = "generated_files" # Base directory for all generated files (preprocessed data, chunks, training results, etc.)
    preprocessed_chunks_count: int = 6 # Number of chunks to split the preprocessed data into for training/testing/validation, should be >= 3 to allow for separate training/testing/validation sets

class PreprocessingSettings(BaseModel):
    """Settings for data loading, cleaning, and augmentation."""
    execute_preprocessing: bool = True # Whether to run the preprocessing step at all
    keep_preprocessed_types: bool = True # Whether to keep preprocessed types, if True, it'll use the last run's files if available
    keep_preprocessed_chunks: bool = True # Whether to keep preprocessed chunks, if True, it'll use the last run's files if available

    # Columns to remove (non-feature data like IPs, timestamps)
    columns_to_drop: List[str] = [
        "Unnamed: 0",
        "Flow ID",
        "Timestamp",
        "Source IP",
        "Destination IP",
        "Source Port",
        "Destination Port",
        "Protocol",
        "Inbound",
        "SimillarHTTP",
        "Fwd Header Length.1"
    ]
    dataset_2019_path: str = "csvs/CIC-DDoS2019" # Path to 2019 dataset, used for oversampling BENIGN samples if enabled in config
    use_2017_benign: bool = True # Whether to use BENIGN samples from 2017 dataset for oversampling
    dataset_2017_path: str = "csvs/CIC-IDS2017" # Path to 2017 dataset, used for oversampling BENIGN samples if enabled in config
    
    # Undersampling settings
    undersample_attacks: bool = True # Whether to undersample 
    multiply_benign: bool = True # For binary classification, BENIGN count = n_attack_types * samples_per_type
    # Strategy: "per_type" keeps fixed samples per attack type; "benign_as_limit" balances to BENIGN count
    undersample_strategy: str = "per_type"  # "per_type" or "benign_as_limit"
    undersample_samples_per_type: int = 1000000

    # Oversampling settings
    synth_data: bool = False
    # "UDP-lag", "Portmap"
    synth_labels: List[str] = ["WebDDoS", "UDPLag"]
    synth_target: int = 5000

class TrainingSettings(BaseModel):
    primary_metric: str = "MCC"
    hyperparameter_ranges: dict = {
        "loss": ["perceptron", "squared_hinge"],
        "alpha": [1e-06, 1e-05, 0.00005, 0.0001, 0.0002]
    }

class AppConfig(BaseModel):
    general: GeneralSettings = GeneralSettings()
    preprocessing: PreprocessingSettings = PreprocessingSettings()
    training: TrainingSettings = TrainingSettings()

config = AppConfig()