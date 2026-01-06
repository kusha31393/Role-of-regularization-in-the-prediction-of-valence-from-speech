"""
Data loading and preprocessing for MSP-Podcast dataset.
Handles feature extraction, normalization, and speaker-aware data splitting.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import h5py
import pickle
from pathlib import Path


class MSPPodcastDataset(Dataset):
    """
    Dataset class for MSP-Podcast corpus.
    
    Expected data format:
    - Features: (N, 6373) numpy array or HDF5 file
    - Labels: DataFrame with columns ['valence', 'arousal', 'dominance', 'speaker_id']
    - Speaker info: DataFrame with speaker metadata
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: pd.DataFrame,
        attribute: str = 'valence',
        transform: Optional[Any] = None
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix (N, feature_dim)
            labels: DataFrame with emotion labels and speaker info
            attribute: Target emotion attribute ('valence', 'arousal', 'dominance')
            transform: Optional feature transformation
        """
        self.features = features
        self.labels = labels
        self.attribute = attribute
        self.transform = transform
        
        assert len(features) == len(labels), "Features and labels must have same length"
        assert attribute in labels.columns, f"Attribute {attribute} not found in labels"
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (features, target) as tensors
        """
        # Get features
        features = self.features[idx].copy()
        
        # Apply transform if provided
        if self.transform is not None:
            features = self.transform(features)
        
        # Get target value
        target = self.labels.iloc[idx][self.attribute]
        
        return torch.FloatTensor(features), torch.FloatTensor([target])


class FeatureStandardizer:
    """
    Feature standardization with outlier clipping as described in the paper.
    """
    
    def __init__(self, max_std_dev: float = 3.0):
        """
        Initialize standardizer.
        
        Args:
            max_std_dev: Maximum standard deviations before clipping
        """
        self.max_std_dev = max_std_dev
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, features: np.ndarray) -> 'FeatureStandardizer':
        """
        Fit the standardizer on training data.
        
        Args:
            features: Training features (N, feature_dim)
        
        Returns:
            Self for chaining
        """
        self.scaler.fit(features)
        self.is_fitted = True
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters.
        
        Args:
            features: Features to transform
        
        Returns:
            Standardized and clipped features
        """
        if not self.is_fitted:
            raise ValueError("Standardizer must be fitted before transform")
        
        # Standardize
        features_std = self.scaler.transform(features)
        
        # Clip outliers beyond max_std_dev standard deviations
        features_clipped = np.clip(
            features_std, 
            -self.max_std_dev, 
            self.max_std_dev
        )
        
        return features_clipped
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            features: Features to fit and transform
        
        Returns:
            Standardized and clipped features
        """
        return self.fit(features).transform(features)


def load_msp_podcast_data(data_dir: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load MSP-Podcast dataset from files.
    
    Expected files in data_dir:
    - features.h5 or features.npy: Feature matrix
    - labels.csv: Emotion labels with speaker info
    
    Args:
        data_dir: Directory containing dataset files
    
    Returns:
        Tuple of (features, labels)
    """
    data_path = Path(data_dir)
    
    # Load features
    feature_file_h5 = data_path / "features.h5"
    feature_file_npy = data_path / "features.npy"
    
    if feature_file_h5.exists():
        with h5py.File(feature_file_h5, 'r') as f:
            features = f['features'][:]
    elif feature_file_npy.exists():
        features = np.load(feature_file_npy)
    else:
        raise FileNotFoundError(f"No feature file found in {data_dir}")
    
    # Load labels
    labels_file = data_path / "labels.csv"
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    labels = pd.read_csv(labels_file)
    
    # Verify required columns
    required_cols = ['valence', 'arousal', 'dominance', 'speaker_id']
    missing_cols = [col for col in required_cols if col not in labels.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in labels: {missing_cols}")
    
    print(f"Loaded features: {features.shape}")
    print(f"Loaded labels: {labels.shape}")
    
    return features, labels


def create_speaker_splits(
    labels: pd.DataFrame,
    train_speakers: Optional[List[str]] = None,
    val_speakers: Optional[List[str]] = None,
    test_speakers: Optional[List[str]] = None
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create speaker-aware data splits.
    
    Args:
        labels: DataFrame with speaker_id column
        train_speakers: List of speaker IDs for training (optional)
        val_speakers: List of speaker IDs for validation (optional)
        test_speakers: List of speaker IDs for testing (optional)
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    if 'speaker_id' not in labels.columns:
        raise ValueError("speaker_id column not found in labels")
    
    # Get indices for each split
    if train_speakers is not None:
        train_indices = labels[labels['speaker_id'].isin(train_speakers)].index.tolist()
    else:
        train_indices = []
    
    if val_speakers is not None:
        val_indices = labels[labels['speaker_id'].isin(val_speakers)].index.tolist()
    else:
        val_indices = []
    
    if test_speakers is not None:
        test_indices = labels[labels['speaker_id'].isin(test_speakers)].index.tolist()
    else:
        test_indices = []
    
    return train_indices, val_indices, test_indices


def create_speaker_dependent_split(
    labels: pd.DataFrame,
    test_speakers: List[str],
    val_speakers: List[str],
    fraction: float = 0.5
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create speaker-dependent split for analysis.
    Adds fraction of test speaker data to training set.
    
    Args:
        labels: DataFrame with speaker_id column
        test_speakers: List of test speaker IDs
        val_speakers: List of validation speaker IDs  
        fraction: Fraction of test speaker data to add to training
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Get all speakers
    all_speakers = labels['speaker_id'].unique()
    train_speakers = [s for s in all_speakers 
                     if s not in test_speakers and s not in val_speakers]
    
    # Start with speaker-independent split
    train_indices, val_indices, test_indices = create_speaker_splits(
        labels, train_speakers, val_speakers, test_speakers
    )
    
    # Add fraction of test speaker data to training
    additional_train_indices = []
    for speaker in test_speakers:
        speaker_indices = labels[labels['speaker_id'] == speaker].index.tolist()
        n_samples = len(speaker_indices)
        n_add = int(n_samples * fraction)
        
        # Add first n_add samples to training
        additional_train_indices.extend(speaker_indices[:n_add])
        
        # Remove those samples from test set
        test_indices = [idx for idx in test_indices if idx not in speaker_indices[:n_add]]
    
    train_indices.extend(additional_train_indices)
    
    return train_indices, val_indices, test_indices


def create_data_loaders(
    features: np.ndarray,
    labels: pd.DataFrame,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    attribute: str = 'valence',
    batch_size: int = 32,
    standardize: bool = True,
    max_std_dev: float = 3.0
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[FeatureStandardizer]]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        features: Feature matrix
        labels: Labels DataFrame
        train_indices: Training sample indices
        val_indices: Validation sample indices
        test_indices: Test sample indices
        attribute: Target emotion attribute
        batch_size: Batch size for data loaders
        standardize: Whether to standardize features
        max_std_dev: Maximum standard deviations for clipping
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Create standardizer if requested
    scaler = None
    if standardize:
        scaler = FeatureStandardizer(max_std_dev=max_std_dev)
        scaler.fit(features[train_indices])
        
        # Transform all features
        features = scaler.transform(features)
    
    # Create datasets
    train_dataset = MSPPodcastDataset(
        features[train_indices], 
        labels.iloc[train_indices].reset_index(drop=True),
        attribute=attribute
    )
    
    val_dataset = MSPPodcastDataset(
        features[val_indices],
        labels.iloc[val_indices].reset_index(drop=True), 
        attribute=attribute
    )
    
    test_dataset = MSPPodcastDataset(
        features[test_indices],
        labels.iloc[test_indices].reset_index(drop=True),
        attribute=attribute
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, scaler


def load_esd_opensmile_data(data_dir: str = "/Users/k.sridhara.murthy/Documents/kusha_research/paper1/data", max_samples: int = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load ESD dataset with OpenSMILE features.
    
    Args:
        data_dir: Path to directory containing processed ESD data
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        Tuple of (features, labels) where:
        - features: (N, 6373) OpenSMILE ComParE features
        - labels: DataFrame with valence, arousal, dominance, speaker_id
    """
    data_path = Path(data_dir)
    
    # Load OpenSMILE features
    opensmile_features_path = data_path / 'opensmile_features_only.pkl'
    if not opensmile_features_path.exists():
        raise FileNotFoundError(f"OpenSMILE features not found at {opensmile_features_path}")
    
    with open(opensmile_features_path, 'rb') as f:
        features = pickle.load(f)
    
    # Load unified dataset for labels
    unified_dataset_path = data_path / 'esd_unified_dataset.pkl'
    if not unified_dataset_path.exists():
        raise FileNotFoundError(f"Unified dataset not found at {unified_dataset_path}")
    
    unified_df = pd.read_pickle(unified_dataset_path)
    
    # Extract labels and speaker information
    # The dimensional_labels column contains dictionaries with A-V-D values
    dimensional_data = unified_df['dimensional_labels'].apply(pd.Series)
    
    labels = pd.DataFrame({
        'valence': dimensional_data['valence'],
        'arousal': dimensional_data['arousal'], 
        'dominance': dimensional_data['dominance'],
        'speaker_id': unified_df['file_path'].str.extract(r'/(\d{4})/')[0]  # Extract speaker ID from path
    })
    
    print(f"Loaded ESD dataset: {len(features)} samples, {features.shape[1]} OpenSMILE features")
    print(f"Speakers: {sorted(labels['speaker_id'].unique())}")
    print(f"Feature range - Valence: [{labels['valence'].min():.3f}, {labels['valence'].max():.3f}]")
    print(f"Feature range - Arousal: [{labels['arousal'].min():.3f}, {labels['arousal'].max():.3f}]")
    print(f"Feature range - Dominance: [{labels['dominance'].min():.3f}, {labels['dominance'].max():.3f}]")
    
    return features.astype(np.float32), labels


def create_dummy_data(
    n_samples: int = 1000,
    feature_dim: int = 6373,
    n_speakers: int = 20,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create dummy data for testing (when real dataset is not available).
    
    Args:
        n_samples: Number of samples to generate
        feature_dim: Feature dimensionality
        n_speakers: Number of speakers
        save_path: Optional path to save dummy data
    
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(42)
    
    # Generate random features
    features = np.random.randn(n_samples, feature_dim)
    
    # Generate random labels with some correlation structure
    base_emotion = np.random.randn(n_samples)
    
    labels = pd.DataFrame({
        'valence': base_emotion + 0.3 * np.random.randn(n_samples),
        'arousal': base_emotion + 0.5 * np.random.randn(n_samples), 
        'dominance': base_emotion + 0.4 * np.random.randn(n_samples),
        'speaker_id': [f'speaker_{i % n_speakers:03d}' for i in range(n_samples)]
    })
    
    # Normalize to reasonable ranges
    for attr in ['valence', 'arousal', 'dominance']:
        labels[attr] = (labels[attr] - labels[attr].mean()) / labels[attr].std()
        labels[attr] = np.clip(labels[attr], -3, 3)  # Clip to 7-point scale range
    
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(save_dir / "features.npy", features)
        labels.to_csv(save_dir / "labels.csv", index=False)
        
        print(f"Dummy data saved to {save_path}")
    
    return features, labels


# Example data loading configuration from the paper
def get_paper_data_splits(labels: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Get speaker splits that match the paper's configuration.
    
    Paper mentions:
    - Test: 50 speakers, 6,069 samples  
    - Validation: 15 speakers, 2,226 samples
    - Training: Remaining speakers and samples
    
    Args:
        labels: Labels DataFrame with speaker_id
    
    Returns:
        Tuple of (train_speakers, val_speakers, test_speakers)
    """
    all_speakers = sorted(labels['speaker_id'].unique())
    
    # Sort speakers by number of samples (for reproducible splitting)
    speaker_counts = labels['speaker_id'].value_counts()
    speakers_by_count = speaker_counts.index.tolist()
    
    # Assign speakers to splits to approximately match paper statistics
    # This is an approximation since we don't have the exact split
    n_speakers = len(all_speakers)
    
    test_speakers = speakers_by_count[:50] if len(speakers_by_count) >= 50 else speakers_by_count[:n_speakers//3]
    val_speakers = speakers_by_count[50:65] if len(speakers_by_count) >= 65 else speakers_by_count[len(test_speakers):len(test_speakers)+min(15, n_speakers//4)]
    train_speakers = [s for s in all_speakers if s not in test_speakers and s not in val_speakers]
    
    return train_speakers, val_speakers, test_speakers