#!/usr/bin/env python3
"""
OpenSMILE ComParE Feature Extraction Script
Extracts ComParE 2016 functional features from ESD dataset for speech emotion recognition

USAGE:
    python src/extract_opensmile_features.py [OPTIONS]

DESCRIPTION:
    This script extracts OpenSMILE ComParE 2016 functional features from audio files.
    ComParE (Computational Paralinguistics Challenge) features are standardized
    acoustic features widely used in emotion recognition research.
    
    Features extracted:
    - 6,373 ComParE 2016 functional features
    - Includes energy, spectral, cepstral, voicing, and temporal features
    - Standardized feature set for reproducible research

OPTIONS:
    --data_dir PATH         Path to ESD dataset directory
    --output_dir PATH       Output directory for processed features
    --speakers LIST         Comma-separated speaker IDs (default: English speakers 0011-0020)
    --max_files_per_emotion Maximum files per emotion category (default: 50)

PREREQUISITES:
    1. Activate virtual environment: source paper1_venv/bin/activate
    2. Install OpenSMILE: pip install opensmile
    3. Ensure ESD dataset is available

OUTPUT FILES:
    - esd_opensmile_features.pkl: Complete feature dataset
    - esd_opensmile_features.csv: Human-readable format
    - feature_metadata.json: Feature extraction metadata

EXAMPLE:
    # Extract features for all English speakers
    python src/extract_opensmile_features.py
    
    # Extract features for specific speakers
    python src/extract_opensmile_features.py --speakers "0011,0012,0013"
    
    # Custom paths and limits
    python src/extract_opensmile_features.py --data_dir /path/to/ESD --max_files_per_emotion 25

INTEGRATION:
    This script can be used standalone or integrated with:
    1. preprocess_esd.py → for complete preprocessing pipeline
    2. train.py → for direct feature-based training
    
NOTES:
    - ComParE features are production-quality features for SER
    - Much more comprehensive than librosa features (6,373 vs ~384 features)
    - Compatible with existing preprocessing and training pipeline
    - Features are automatically saved for later use in training
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import pickle
from datetime import datetime
import opensmile

warnings.filterwarnings('ignore')

# Emotion categories in ESD dataset
EMOTION_CATEGORIES = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

class OpenSMILEFeatureExtractor:
    """OpenSMILE ComParE feature extractor for ESD dataset"""
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize OpenSMILE feature extractor
        
        Args:
            data_dir: Path to ESD dataset
            output_dir: Path to save extracted features
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenSMILE with ComParE 2016 configuration
        print("Initializing OpenSMILE with ComParE 2016 configuration...")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        # Get feature names and metadata
        self.feature_names = None
        self.feature_dim = None
        self._initialize_feature_info()
        
    def _initialize_feature_info(self):
        """Initialize feature metadata by processing a dummy file"""
        # Find any audio file to get feature information
        sample_file = None
        for speaker_dir in self.data_dir.glob('*'):
            if speaker_dir.is_dir():
                for emotion_dir in speaker_dir.glob('*'):
                    if emotion_dir.is_dir():
                        audio_files = list(emotion_dir.glob('*.wav'))
                        if audio_files:
                            sample_file = audio_files[0]
                            break
                if sample_file:
                    break
        
        if sample_file:
            try:
                # Extract features from sample file to get metadata
                features = self.smile.process_file(str(sample_file))
                self.feature_names = features.columns.tolist()
                self.feature_dim = len(self.feature_names)
                print(f"ComParE 2016 features initialized: {self.feature_dim} features")
            except Exception as e:
                print(f"Warning: Could not initialize feature metadata: {e}")
                self.feature_names = None
                self.feature_dim = 6373  # Default ComParE 2016 feature count
        else:
            print("Warning: No audio files found for feature initialization")
            self.feature_dim = 6373  # Default ComParE 2016 feature count
    
    def extract_features_from_file(self, audio_path: str) -> np.ndarray:
        """
        Extract ComParE features from a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector (6373 features for ComParE 2016)
        """
        try:
            features = self.smile.process_file(audio_path)
            return features.values.flatten()
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            # Return zero vector with correct dimensionality
            return np.zeros(self.feature_dim or 6373)
    
    def process_speaker(self, speaker_id: str, max_files_per_emotion: int = 50) -> List[Dict]:
        """
        Process all audio files for a speaker
        
        Args:
            speaker_id: Speaker ID (e.g., '0011')
            max_files_per_emotion: Maximum files to process per emotion
            
        Returns:
            List of feature dictionaries
        """
        speaker_dir = self.data_dir / speaker_id
        if not speaker_dir.exists():
            print(f"Warning: Speaker {speaker_id} directory does not exist")
            return []
        
        data = []
        
        # Process each emotion folder
        for emotion in EMOTION_CATEGORIES:
            emotion_dir = speaker_dir / emotion
            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} does not exist")
                continue
            
            # Get audio files
            audio_files = list(emotion_dir.glob('*.wav'))[:max_files_per_emotion]
            
            for audio_file in tqdm(audio_files, 
                                 desc=f"Processing {speaker_id}/{emotion}", 
                                 leave=False):
                # Extract ComParE features
                features = self.extract_features_from_file(str(audio_file))
                
                # Create data entry
                entry = {
                    'file_path': str(audio_file),
                    'speaker_id': speaker_id,
                    'emotion': emotion,
                    'features': features,
                    'feature_extraction_method': 'OpenSMILE_ComParE_2016',
                    'feature_dim': len(features)
                }
                data.append(entry)
        
        return data
    
    def extract_features_for_unified_dataset(self) -> pd.DataFrame:
        """
        Extract ComParE features for files in the existing unified dataset
        Updates the unified dataset and creates features-only files
        
        Returns:
            DataFrame with extracted features in unified dataset order
        """
        # Load the existing unified dataset
        unified_dataset_path = self.output_dir / 'esd_unified_dataset.csv'
        if not unified_dataset_path.exists():
            print(f"Error: Unified dataset not found at {unified_dataset_path}")
            return pd.DataFrame()
        
        print(f"Loading existing unified dataset from: {unified_dataset_path}")
        unified_df = pd.read_csv(unified_dataset_path)
        
        print(f"Found {len(unified_df)} files in unified dataset")
        print(f"Processing OpenSMILE feature extraction...")
        
        # Extract features for each file in the unified dataset order
        features_data = []
        
        for idx, row in tqdm(unified_df.iterrows(), total=len(unified_df), 
                           desc="Extracting OpenSMILE features"):
            audio_path = row['file_path']
            
            # Extract ComParE features
            features = self.extract_features_from_file(audio_path)
            
            # Store features data in same order as unified dataset
            features_data.append({
                'file_path': audio_path,
                'features': features,
                'feature_dim': len(features)
            })
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_data)
        
        # Update the unified dataset with OpenSMILE features
        print("Updating unified dataset with OpenSMILE features...")
        unified_df['opensmile_features'] = features_df['features'].tolist()
        
        # Save updated unified dataset
        updated_unified_path = self.output_dir / 'esd_unified_dataset_with_opensmile.csv'
        updated_unified_pkl = self.output_dir / 'esd_unified_dataset_with_opensmile.pkl'
        
        # For CSV, we need to convert features to string representation
        unified_csv = unified_df.copy()
        unified_csv['opensmile_features'] = unified_csv['opensmile_features'].apply(
            lambda x: ','.join(map(str, x)) if isinstance(x, np.ndarray) else x
        )
        unified_csv.to_csv(updated_unified_path, index=False)
        
        # Save as pickle to preserve numpy arrays
        unified_df.to_pickle(updated_unified_pkl)
        
        print(f"Updated unified dataset saved to:")
        print(f"  - CSV: {updated_unified_path}")
        print(f"  - Pickle: {updated_unified_pkl}")
        
        print(f"\nFeature extraction complete!")
        print(f"Total files processed: {len(features_df)}")
        print(f"Feature dimensionality: {features_df.iloc[0]['feature_dim']} (ComParE 2016)")
        
        return features_df
    
    def extract_features_dataset(self, 
                                speakers: Optional[List[str]] = None,
                                max_files_per_emotion: int = 50) -> pd.DataFrame:
        """
        Extract ComParE features for specified speakers (legacy method)
        
        Args:
            speakers: List of speaker IDs. If None, uses English speakers (0011-0020)
            max_files_per_emotion: Maximum files to process per emotion
            
        Returns:
            DataFrame with extracted features
        """
        if speakers is None:
            # Default to English speakers
            speakers = [f'{i:04d}' for i in range(11, 21)]
        
        all_data = []
        total_files = 0
        
        print(f"Extracting ComParE features for speakers: {speakers}")
        print(f"Maximum {max_files_per_emotion} files per emotion per speaker")
        
        for speaker in speakers:
            speaker_data = self.process_speaker(speaker, max_files_per_emotion)
            all_data.extend(speaker_data)
            print(f"Speaker {speaker}: {len(speaker_data)} files processed")
            total_files += len(speaker_data)
        
        if not all_data:
            print("Warning: No data extracted!")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        print(f"\nFeature extraction complete!")
        print(f"Total files processed: {total_files}")
        print(f"Feature dimensionality: {df.iloc[0]['feature_dim']} (ComParE 2016)")
        print(f"Speakers: {len(set(df['speaker_id']))}")
        print(f"Emotions per speaker: {[emotion for emotion in EMOTION_CATEGORIES]}")
        
        return df
    
    def save_features_only(self, features_df: pd.DataFrame, filename_prefix: str = "opensmile_features_only"):
        """
        Save only the features in the order of the unified dataset
        
        Args:
            features_df: DataFrame with features in unified dataset order
            filename_prefix: Prefix for output files
        """
        if features_df.empty:
            print("Warning: No features to save!")
            return
        
        # Extract features array
        features_array = np.vstack(features_df['features'].values)
        
        # Save features-only as pickle (preserves numpy array format)
        features_pkl = self.output_dir / f"{filename_prefix}.pkl"
        with open(features_pkl, 'wb') as f:
            pickle.dump(features_array, f)
        print(f"Features-only pickle saved to: {features_pkl}")
        
        # Save features-only as CSV
        if self.feature_names:
            feature_columns = self.feature_names
        else:
            feature_columns = [f'feature_{i:04d}' for i in range(features_array.shape[1])]
        
        features_csv_df = pd.DataFrame(features_array, columns=feature_columns)
        features_csv = self.output_dir / f"{filename_prefix}.csv"
        features_csv_df.to_csv(features_csv, index=False)
        print(f"Features-only CSV saved to: {features_csv}")
        
        # Save metadata
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'feature_set': 'ComParE_2016',
            'feature_level': 'Functionals',
            'feature_dim': int(features_array.shape[1]),
            'total_samples': len(features_array),
            'note': 'Features extracted in the same order as esd_unified_dataset.csv',
            'feature_extraction_method': 'OpenSMILE_ComParE_2016'
        }
        
        if self.feature_names:
            metadata['feature_names'] = self.feature_names
        
        metadata_file = self.output_dir / f"{filename_prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Features-only metadata saved to: {metadata_file}")

    def save_features(self, df: pd.DataFrame, filename_prefix: str = "esd_opensmile_features"):
        """
        Save extracted features to files (legacy method)
        
        Args:
            df: DataFrame with extracted features
            filename_prefix: Prefix for output files
        """
        if df.empty:
            print("Warning: No data to save!")
            return
        
        # Save as pickle (preserves numpy arrays)
        pickle_file = self.output_dir / f"{filename_prefix}.pkl"
        df.to_pickle(pickle_file)
        print(f"Features saved to: {pickle_file}")
        
        # Save features as CSV (for inspection)
        csv_data = df.drop('features', axis=1).copy()
        
        # Add flattened features as separate columns
        features_array = np.vstack(df['features'].values)
        
        # Create feature column names
        if self.feature_names:
            feature_columns = self.feature_names
        else:
            feature_columns = [f'feature_{i:04d}' for i in range(features_array.shape[1])]
        
        # Add features to CSV data
        feature_df = pd.DataFrame(features_array, columns=feature_columns)
        csv_data = pd.concat([csv_data.reset_index(drop=True), feature_df], axis=1)
        
        csv_file = self.output_dir / f"{filename_prefix}.csv"
        csv_data.to_csv(csv_file, index=False)
        print(f"CSV data saved to: {csv_file}")
        
        # Save metadata
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'feature_set': 'ComParE_2016',
            'feature_level': 'Functionals',
            'feature_dim': int(features_array.shape[1]),
            'total_samples': len(df),
            'feature_extraction_method': 'OpenSMILE_ComParE_2016'
        }
        
        if hasattr(df, 'speaker_id') and 'speaker_id' in df.columns:
            metadata['speakers'] = sorted(df['speaker_id'].unique().tolist())
        if hasattr(df, 'emotion') and 'emotion' in df.columns:
            metadata['emotions'] = sorted(df['emotion'].unique().tolist())
            if hasattr(df, 'speaker_id') and 'speaker_id' in df.columns:
                metadata['files_per_emotion_per_speaker'] = {f"{k[0]}_{k[1]}": v for k, v in df.groupby(['speaker_id', 'emotion']).size().to_dict().items()}
        
        if self.feature_names:
            metadata['feature_names'] = self.feature_names
        
        metadata_file = self.output_dir / f"{filename_prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_file}")
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute feature statistics
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with statistics
        """
        if df.empty:
            return {}
        
        features_array = np.vstack(df['features'].values)
        
        stats = {
            'feature_statistics': {
                'mean': np.mean(features_array, axis=0),
                'std': np.std(features_array, axis=0),
                'min': np.min(features_array, axis=0),
                'max': np.max(features_array, axis=0),
                'median': np.median(features_array, axis=0)
            },
            'dataset_statistics': {
                'total_samples': len(df),
                'samples_per_speaker': df['speaker_id'].value_counts().to_dict(),
                'samples_per_emotion': df['emotion'].value_counts().to_dict()
            }
        }
        
        return stats


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract OpenSMILE ComParE features from ESD dataset'
    )
    parser.add_argument('--data_dir', type=str, 
                       default='/Users/k.sridhara.murthy/Documents/kusha_research/data/ESD',
                       help='Path to ESD dataset')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/k.sridhara.murthy/Documents/kusha_research/paper1/data',
                       help='Output directory for extracted features')
    parser.add_argument('--use_unified_dataset', action='store_true',
                       help='Extract features for existing unified dataset (recommended)')
    parser.add_argument('--speakers', type=str,
                       default='0011,0012,0013,0014,0015,0016,0017,0018,0019,0020',
                       help='Comma-separated speaker IDs (for legacy mode)')
    parser.add_argument('--max_files_per_emotion', type=int, default=50,
                       help='Maximum files to process per emotion (for legacy mode)')
    parser.add_argument('--output_prefix', type=str, default='esd_opensmile_features',
                       help='Prefix for output files (for legacy mode)')
    
    args = parser.parse_args()
    
    # Create feature extractor
    print("=" * 70)
    print("OpenSMILE ComParE Feature Extraction for ESD Dataset")
    print("=" * 70)
    
    extractor = OpenSMILEFeatureExtractor(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    if args.use_unified_dataset:
        # NEW: Process unified dataset (Option 1 - Recommended)
        print("Processing files from existing unified dataset...")
        features_df = extractor.extract_features_for_unified_dataset()
        
        if not features_df.empty:
            # Save features-only files in unified dataset order
            print("\nSaving features-only files...")
            extractor.save_features_only(features_df, "opensmile_features_only")
            
            print(f"\n" + "=" * 70)
            print("SUCCESS: OpenSMILE features extracted and integrated!")
            print(f"\nFiles created:")
            print(f"  1. esd_unified_dataset_with_opensmile.pkl (updated unified dataset)")
            print(f"  2. esd_unified_dataset_with_opensmile.csv (updated unified dataset)")
            print(f"  3. opensmile_features_only.pkl (features-only in unified order)")
            print(f"  4. opensmile_features_only.csv (features-only in unified order)")
            print(f"  5. opensmile_features_only_metadata.json (extraction metadata)")
            print(f"\nTotal files processed: {len(features_df)}")
            print(f"Feature dimension: 6,373 (ComParE 2016)")
            print("=" * 70)
        else:
            print("Error: No features extracted from unified dataset!")
    
    else:
        # LEGACY: Extract by speakers (original method)
        speakers = [s.strip() for s in args.speakers.split(',')]
        
        print(f"Using legacy mode - extracting features for speakers: {speakers}")
        df = extractor.extract_features_dataset(
            speakers=speakers,
            max_files_per_emotion=args.max_files_per_emotion
        )
        
        if not df.empty:
            # Save features
            print(f"\nSaving features...")
            extractor.save_features(df, args.output_prefix)
            
            # Compute and display statistics
            stats = extractor.get_feature_statistics(df)
            print(f"\nDataset Summary:")
            print(f"  Total samples: {stats['dataset_statistics']['total_samples']}")
            print(f"  Feature dimension: {df.iloc[0]['feature_dim']}")
            print(f"  Speakers: {len(stats['dataset_statistics']['samples_per_speaker'])}")
            print(f"  Emotions: {len(stats['dataset_statistics']['samples_per_emotion'])}")
            
            print(f"\nSamples per emotion:")
            for emotion, count in stats['dataset_statistics']['samples_per_emotion'].items():
                print(f"  {emotion}: {count}")
            
            print(f"\nSamples per speaker:")
            for speaker, count in stats['dataset_statistics']['samples_per_speaker'].items():
                print(f"  {speaker}: {count}")
            
            print(f"\n" + "=" * 70)
            print("Feature extraction completed successfully!")
            print(f"Output files saved to: {args.output_dir}")
            print("=" * 70)
        
        else:
            print("Error: No features extracted!")


if __name__ == '__main__':
    main()