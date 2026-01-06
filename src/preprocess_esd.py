#!/usr/bin/env python3
"""
ESD Dataset Preprocessing Script
Extracts acoustic features and integrates with dimensional emotion labels for model training

USAGE:
    python src/preprocess_esd.py [OPTIONS]

DESCRIPTION:
    This script processes ESD dataset to create training-ready data by:
    1. Loading dimensional labels from extract_w2v2_emotions.py output
    2. Extracting acoustic features (OpenSMILE/librosa) from audio files
    3. Creating enhanced dataset with both categorical and dimensional labels
    4. Preparing speaker-aware train/val/test splits for experiments

OPTIONS:
    --data_dir PATH         Path to ESD dataset directory (default: ~/Documents/kusha_research/data/ESD)
    --output_dir PATH       Output directory for processed data (default: ./data)
    --use_opensmile         Use OpenSMILE features if available (default: librosa fallback)

PREREQUISITES:
    1. Activate virtual environment: source paper1_venv/bin/activate
    2. Run extract_w2v2_emotions.py first to generate dimensional labels
    3. Install dependencies: pip install scikit-learn pandas numpy librosa tqdm
    4. Optional: Install OpenSMILE for better acoustic features

INPUT FILES (from extract_w2v2_emotions.py):
    - esd_unified_dataset.pkl: Dataset with dimensional labels

OUTPUT FILES:
    - esd_english_processed_with_dimensional.pkl: Complete enhanced dataset
    - esd_english_splits.pkl: Training splits ready for experiments
    - label_encoder.pkl: Categorical emotion encoder

EXAMPLE:
    # Basic usage with librosa features
    python src/preprocess_esd.py
    
    # With OpenSMILE features (if installed)
    python src/preprocess_esd.py --use_opensmile
    
    # Custom paths
    python src/preprocess_esd.py --data_dir /path/to/ESD --output_dir ./results

WORKFLOW:
    1. extract_w2v2_emotions.py → extracts dimensional labels
    2. preprocess_esd.py → adds acoustic features + creates training splits
    3. train.py → runs regularization experiments

NOTES:
    - Processes English speakers (0011-0020) only
    - Creates speaker-aware splits: 6 train, 2 val, 2 test speakers
    - Enhanced dataset includes: features, categorical labels, V-A-D scores
    - Ready for dropout regularization experiments from the research paper
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import librosa
import warnings
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Emotion categories in ESD dataset
EMOTION_CATEGORIES = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

class ESDPreprocessor:
    """Preprocessor for ESD dataset - English samples only"""
    
    def __init__(self, data_dir: str, output_dir: str, use_opensmile: bool = False):
        """
        Initialize preprocessor
        
        Args:
            data_dir: Path to ESD dataset
            output_dir: Path to save processed data
            use_opensmile: Whether to use OpenSMILE (if installed) or librosa
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_opensmile = use_opensmile
        
        # Initialize label encoder for categorical emotions
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(EMOTION_CATEGORIES)
        
        # Check if OpenSMILE is available
        if self.use_opensmile:
            self.opensmile_available = self._check_opensmile()
            if not self.opensmile_available:
                print("OpenSMILE not found. Falling back to librosa features.")
                self.use_opensmile = False
    
    def _check_opensmile(self) -> bool:
        """Check if OpenSMILE is installed"""
        try:
            import opensmile
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            print("OpenSMILE found and initialized.")
            return True
        except ImportError:
            return False
    
    def extract_librosa_features(self, audio_path: str) -> np.ndarray:
        """
        Extract acoustic features using librosa (alternative to OpenSMILE)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            features = []
            
            # 1. MFCCs (13 coefficients + deltas + delta-deltas)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # 2. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # 4. Energy features
            rms = librosa.feature.rms(y=y)
            
            # 5. Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # 6. Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Compute statistics for each feature (mean, std, min, max, etc.)
            for feat in [mfccs, mfccs_delta, mfccs_delta2]:
                features.extend([
                    np.mean(feat, axis=1), np.std(feat, axis=1),
                    np.min(feat, axis=1), np.max(feat, axis=1),
                    np.median(feat, axis=1)
                ])
            
            for feat in [spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, rms]:
                features.extend([
                    np.mean(feat), np.std(feat), np.min(feat), 
                    np.max(feat), np.median(feat)
                ])
            
            for feat in spectral_contrast:
                features.extend([
                    np.mean(feat, axis=1), np.std(feat, axis=1),
                    np.min(feat, axis=1), np.max(feat, axis=1)
                ])
            
            # Add tempo
            features.append(tempo)
            
            # Flatten all features
            feature_vector = np.hstack([f.flatten() if isinstance(f, np.ndarray) else f for f in features])
            
            # Pad or truncate to fixed size (384 features as approximation)
            target_size = 384
            if len(feature_vector) > target_size:
                feature_vector = feature_vector[:target_size]
            elif len(feature_vector) < target_size:
                feature_vector = np.pad(feature_vector, (0, target_size - len(feature_vector)))
            
            return feature_vector
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return np.zeros(384)  # Return zero vector on error
    
    def extract_opensmile_features(self, audio_path: str) -> np.ndarray:
        """
        Extract ComParE features using OpenSMILE
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector
        """
        try:
            features = self.smile.process_file(audio_path)
            return features.values.flatten()
        except Exception as e:
            print(f"Error extracting OpenSMILE features from {audio_path}: {e}")
            return self.extract_librosa_features(audio_path)
    
    def process_speaker(self, speaker_id: str) -> pd.DataFrame:
        """
        Process all audio files for a speaker
        
        Args:
            speaker_id: Speaker ID (e.g., '0011')
            
        Returns:
            DataFrame with features and labels
        """
        speaker_dir = self.data_dir / speaker_id
        data = []
        
        # Process each emotion folder
        for emotion in EMOTION_CATEGORIES:
            emotion_dir = speaker_dir / emotion
            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} does not exist")
                continue
            
            # Get categorical label (encoded as integer)
            emotion_label = self.label_encoder.transform([emotion])[0]
            
            # Process each audio file
            audio_files = list(emotion_dir.glob('*.wav'))
            for audio_file in tqdm(audio_files[:50], desc=f"Processing {speaker_id}/{emotion}", leave=False):
                # Extract features
                if self.use_opensmile and self.opensmile_available:
                    features = self.extract_opensmile_features(str(audio_file))
                else:
                    features = self.extract_librosa_features(str(audio_file))
                
                # Create data entry
                entry = {
                    'file_path': str(audio_file),
                    'speaker_id': speaker_id,
                    'emotion': emotion,
                    'emotion_label': emotion_label,
                    'features': features
                }
                data.append(entry)
        
        return pd.DataFrame(data)
    
    def process_english_dataset(self):
        """
        Process the English portion of the ESD dataset (speakers 0011-0020)
        Integrates with dimensional labels from w2v2 emotion extraction
        """
        # Check if unified dataset with dimensional labels exists
        unified_dataset_path = self.output_dir / 'esd_unified_dataset.pkl'
        
        if unified_dataset_path.exists():
            print(f"Found unified dataset with dimensional labels: {unified_dataset_path}")
            print("Loading existing unified dataset and integrating with acoustic features...")
            
            # Load the unified dataset created by extract_w2v2_emotions.py
            unified_df = pd.read_pickle(unified_dataset_path)
            
            # Extract features for all files in the unified dataset
            enhanced_data = []
            
            for idx, row in tqdm(unified_df.iterrows(), total=len(unified_df), desc="Adding acoustic features"):
                audio_path = row['file_path']
                
                # Extract acoustic features (librosa for now, OpenSMILE in next task)
                if self.use_opensmile and self.opensmile_available:
                    features = self.extract_opensmile_features(audio_path)
                else:
                    features = self.extract_librosa_features(audio_path)
                
                # Extract speaker ID from path
                speaker_id = Path(audio_path).parent.parent.name
                
                # Get categorical emotion label
                emotion = row['categorical_label']
                emotion_label = self.label_encoder.transform([emotion])[0]
                
                # Create enhanced entry
                enhanced_entry = {
                    'file_path': audio_path,
                    'speaker_id': speaker_id,
                    'emotion': emotion,
                    'emotion_label': emotion_label,
                    'features': features,
                    'dimensional_labels': row['dimensional_labels'],
                    'arousal': row['dimensional_labels']['arousal'],
                    'valence': row['dimensional_labels']['valence'],
                    'dominance': row['dimensional_labels']['dominance']
                }
                enhanced_data.append(enhanced_entry)
            
            full_dataset = pd.DataFrame(enhanced_data)
            
        else:
            print("Unified dataset not found. Creating from scratch...")
            # Fall back to original processing
            speakers = [f'{i:04d}' for i in range(11, 21)]
            all_data = []
            
            for speaker in speakers:
                speaker_path = self.data_dir / speaker
                if speaker_path.exists():
                    print(f"\nProcessing speaker {speaker}...")
                    speaker_data = self.process_speaker(speaker)
                    all_data.append(speaker_data)
                else:
                    print(f"Warning: Speaker {speaker} directory not found")
            
            full_dataset = pd.concat(all_data, ignore_index=True)
        
        # Save enhanced dataset
        output_file = self.output_dir / 'esd_english_processed_with_dimensional.pkl'
        full_dataset.to_pickle(output_file)
        print(f"\nEnhanced dataset saved to {output_file}")
        
        # Save label encoder for later use
        encoder_file = self.output_dir / 'label_encoder.pkl'
        with open(encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to {encoder_file}")
        
        # Prepare training data with both categorical and dimensional labels
        self.prepare_training_data(full_dataset)
        
        return full_dataset
    
    def prepare_training_data(self, df: pd.DataFrame):
        """
        Prepare data in format expected by training scripts with both categorical and dimensional labels
        
        Args:
            df: Processed dataframe with enhanced structure
        """
        # Extract features and labels
        features = np.vstack(df['features'].values)
        labels = df['emotion_label'].values  # Categorical labels (0-4)
        emotion_names = df['emotion'].values
        speaker_ids = df['speaker_id'].values
        
        # Extract dimensional labels if available
        if 'arousal' in df.columns:
            arousal = df['arousal'].values
            valence = df['valence'].values
            dominance = df['dominance'].values
            dimensional_scores = np.column_stack([arousal, valence, dominance])
        else:
            # Fallback if dimensional labels not available
            arousal = valence = dominance = None
            dimensional_scores = None
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Create speaker-aware train/val/test splits
        # Ensure each split has samples from all speakers
        unique_speakers = np.unique(speaker_ids)
        
        # Split speakers: 6 train, 2 val, 2 test
        np.random.seed(42)
        np.random.shuffle(unique_speakers)
        train_speakers = unique_speakers[:6]
        val_speakers = unique_speakers[6:8]
        test_speakers = unique_speakers[8:10]
        
        # Create masks for each split
        train_mask = np.isin(speaker_ids, train_speakers)
        val_mask = np.isin(speaker_ids, val_speakers)
        test_mask = np.isin(speaker_ids, test_speakers)
        
        # Save splits with both categorical and dimensional labels
        data_splits = {
            'train': {
                'features': features_normalized[train_mask],
                'labels': labels[train_mask],  # Categorical
                'emotion_names': emotion_names[train_mask],
                'speaker_ids': speaker_ids[train_mask]
            },
            'val': {
                'features': features_normalized[val_mask],
                'labels': labels[val_mask],  # Categorical
                'emotion_names': emotion_names[val_mask],
                'speaker_ids': speaker_ids[val_mask]
            },
            'test': {
                'features': features_normalized[test_mask],
                'labels': labels[test_mask],  # Categorical
                'emotion_names': emotion_names[test_mask],
                'speaker_ids': speaker_ids[test_mask]
            },
            'scaler': scaler,
            'label_encoder': self.label_encoder,
            'feature_dim': features.shape[1],
            'num_classes': len(EMOTION_CATEGORIES),
            'emotion_categories': EMOTION_CATEGORIES
        }
        
        # Add dimensional labels if available
        if dimensional_scores is not None:
            data_splits['train']['dimensional_scores'] = dimensional_scores[train_mask]
            data_splits['train']['arousal'] = arousal[train_mask]
            data_splits['train']['valence'] = valence[train_mask]
            data_splits['train']['dominance'] = dominance[train_mask]
            
            data_splits['val']['dimensional_scores'] = dimensional_scores[val_mask]
            data_splits['val']['arousal'] = arousal[val_mask] 
            data_splits['val']['valence'] = valence[val_mask]
            data_splits['val']['dominance'] = dominance[val_mask]
            
            data_splits['test']['dimensional_scores'] = dimensional_scores[test_mask]
            data_splits['test']['arousal'] = arousal[test_mask]
            data_splits['test']['valence'] = valence[test_mask]
            data_splits['test']['dominance'] = dominance[test_mask]
            
            # Add dimensional metadata
            data_splits['has_dimensional'] = True
            data_splits['dimensional_dim'] = 3  # A, V, D
        else:
            data_splits['has_dimensional'] = False
        
        # Save to file
        output_file = self.output_dir / 'esd_english_splits.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(data_splits, f)
        
        print(f"\nTraining data saved to {output_file}")
        print(f"\nDataset statistics:")
        print(f"  - Total samples: {len(features)}")
        print(f"  - Train: {train_mask.sum()} samples from speakers {train_speakers}")
        print(f"  - Val: {val_mask.sum()} samples from speakers {val_speakers}")
        print(f"  - Test: {test_mask.sum()} samples from speakers {test_speakers}")
        print(f"  - Feature dimension: {features.shape[1]}")
        print(f"  - Number of classes: {len(EMOTION_CATEGORIES)}")
        print(f"  - Emotion categories: {EMOTION_CATEGORIES}")
        print(f"  - Has dimensional labels: {data_splits['has_dimensional']}")
        
        # Print class distribution
        print(f"\nClass distribution in splits:")
        for split_name in ['train', 'val', 'test']:
            split_labels = data_splits[split_name]['labels']
            print(f"\n{split_name.capitalize()}:")
            for i, emotion in enumerate(EMOTION_CATEGORIES):
                count = np.sum(split_labels == i)
                percentage = count / len(split_labels) * 100
                print(f"  {emotion}: {count} ({percentage:.1f}%)")
        
        # Print dimensional statistics if available
        if dimensional_scores is not None:
            print(f"\nDimensional emotion statistics:")
            for dim_name, dim_values in [('Arousal', arousal), ('Valence', valence), ('Dominance', dominance)]:
                print(f"\n{dim_name}:")
                print(f"  Mean: {np.mean(dim_values):.3f}")
                print(f"  Std:  {np.std(dim_values):.3f}")
                print(f"  Min:  {np.min(dim_values):.3f}")
                print(f"  Max:  {np.max(dim_values):.3f}")
            
            print(f"\nMean dimensional scores by categorical emotion:")
            for i, emotion in enumerate(EMOTION_CATEGORIES):
                emotion_mask = labels == i
                if np.sum(emotion_mask) > 0:
                    emotion_arousal = np.mean(arousal[emotion_mask])
                    emotion_valence = np.mean(valence[emotion_mask])
                    emotion_dominance = np.mean(dominance[emotion_mask])
                    print(f"  {emotion}: A={emotion_arousal:.3f}, V={emotion_valence:.3f}, D={emotion_dominance:.3f}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess ESD dataset (English samples only)')
    parser.add_argument('--data_dir', type=str, 
                       default='/Users/k.sridhara.murthy/Documents/kusha_research/data/ESD',
                       help='Path to ESD dataset')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/k.sridhara.murthy/Documents/kusha_research/paper1/data',
                       help='Output directory for processed data')
    parser.add_argument('--use_opensmile', action='store_true',
                       help='Use OpenSMILE if available (otherwise use librosa)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    print("Initializing ESD preprocessor for English samples...")
    preprocessor = ESDPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_opensmile=args.use_opensmile
    )
    
    # Process dataset
    print("Processing English speakers (0011-0020)...")
    preprocessor.process_english_dataset()
    
    print("\nPreprocessing complete!")
    print(f"Processed data saved to: {args.output_dir}")
    print("\nYou can now use the categorical labels for initial training,")
    print("and we can later map them to continuous V-A-D values as needed.")


if __name__ == '__main__':
    main()