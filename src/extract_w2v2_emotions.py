#!/usr/bin/env python3
"""
Dimensional Emotion Extraction using wav2vec 2.0 Model
Extracts arousal, valence, and dominance scores from audio files using the w2v2-how-to model

USAGE:
    python src/extract_w2v2_emotions.py [OPTIONS]

DESCRIPTION:
    This script extracts dimensional emotion labels (arousal, valence, dominance) from 
    ESD dataset audio files using a pre-trained wav2vec 2.0 model. It creates a unified
    DataFrame with file paths, categorical labels, and dimensional scores.

OPTIONS:
    --esd_data_dir PATH     Path to ESD dataset directory (default: ~/Documents/kusha_research/data/ESD)
    --output_dir PATH       Output directory for results (default: ./data)
    --cache_dir PATH        Cache directory for model download (default: ./cache)
    --model_dir PATH        Model directory (default: ./models)

PREREQUISITES:
    1. Activate virtual environment: source paper1_venv/bin/activate
    2. Install dependencies: pip install audonnx audeer librosa tqdm pandas
    3. ESD dataset should be organized as: ESD/SPEAKER_ID/EMOTION/*.wav

OUTPUT FILES:
    - esd_unified_dataset.pkl: Complete dataset with dimensional labels
    - esd_unified_dataset.csv: CSV version for inspection

EXAMPLE:
    # Basic usage (processes English speakers 0011-0020)
    python src/extract_w2v2_emotions.py
    
    # Custom paths
    python src/extract_w2v2_emotions.py --esd_data_dir /path/to/ESD --output_dir ./results

NOTES:
    - Processes only English speakers (0011-0020) from ESD dataset
    - Downloads w2v2 model automatically on first run (~500MB)
    - Processes 50 files per emotion category (2,500 total files)
    - Creates unified DataFrame with 4 columns: file_path, opensmile_features, 
      categorical_label, dimensional_labels
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import warnings
from tqdm import tqdm
import pickle
import audeer
import audonnx

warnings.filterwarnings('ignore')

class W2V2EmotionExtractor:
    """Extract dimensional emotions (arousal, valence, dominance) using wav2vec 2.0 model"""
    
    def __init__(self, cache_dir: str = None, model_dir: str = None):
        """
        Initialize the emotion extractor
        
        Args:
            cache_dir: Directory to cache the downloaded model
            model_dir: Directory to store the model
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./cache')
        self.model_dir = Path(model_dir) if model_dir else Path('./model')
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.sampling_rate = 16000
        
    def download_and_load_model(self):
        """Download and load the w2v2 emotion recognition model"""
        try:
            # Check if model already exists
            if (self.model_dir / 'model.onnx').exists() or any(self.model_dir.glob('*.onnx')):
                print("Found existing model, loading...")
                self.model = audonnx.load(self.model_dir)
                print("Model loaded successfully!")
                return True
            
            # Model URL from the w2v2-how-to repository
            url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
            
            print("Model not found. Downloading w2v2 emotion recognition model...")
            archive_path = audeer.download_url(url, self.cache_dir, verbose=True)
            
            print("Extracting model...")
            audeer.extract_archive(archive_path, self.model_dir)
            
            print("Loading model...")
            self.model = audonnx.load(self.model_dir)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_emotions(self, audio_path: str) -> Dict[str, float]:
        """
        Extract arousal, valence, and dominance from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with emotion scores
        """
        try:
            # Load audio at correct sampling rate
            signal, sr = librosa.load(audio_path, sr=self.sampling_rate)
            signal = signal.astype(np.float32)
            
            # Get predictions from model
            result = self.model(signal, self.sampling_rate)
            
            # Extract logits (arousal, dominance, valence)
            if 'logits' in result:
                logits = result['logits']
            else:
                # If logits not directly available, try to extract from result
                logits = result
                if isinstance(logits, dict):
                    # Look for the emotion predictions in the result
                    for key in ['output', 'prediction', 'emotion']:
                        if key in logits:
                            logits = logits[key]
                            break
            
            # Ensure we have the right format
            if isinstance(logits, (list, tuple, np.ndarray)):
                logits = np.array(logits)
                if logits.ndim > 1:
                    logits = logits.flatten()
                
                # The model should output [arousal, dominance, valence]
                if len(logits) >= 3:
                    arousal = float(logits[0])
                    dominance = float(logits[1])
                    valence = float(logits[2])
                else:
                    print(f"Warning: Unexpected logits shape {logits.shape}")
                    arousal = dominance = valence = 0.0
            else:
                print(f"Warning: Unexpected result format: {type(logits)}")
                arousal = dominance = valence = 0.0
            
            return {
                'arousal': arousal,
                'dominance': dominance,
                'valence': valence
            }
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return {'arousal': 0.0, 'dominance': 0.0, 'valence': 0.0}
    
    def process_esd_dataset(self, esd_data_dir: str, output_dir: str):
        """
        Process ESD dataset to create unified DataFrame with file paths, categorical labels, 
        and dimensional emotion labels (OpenSMILE features placeholder for later)
        
        Args:
            esd_data_dir: Path to ESD dataset
            output_dir: Directory to save results
        """
        if self.model is None:
            print("Model not loaded. Please run download_and_load_model() first.")
            return
        
        esd_path = Path(esd_data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process English speakers (0011-0020)
        english_speakers = [f'{i:04d}' for i in range(11, 21)]
        emotion_categories = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        all_results = []
        
        for speaker_id in english_speakers:
            speaker_dir = esd_path / speaker_id
            if not speaker_dir.exists():
                print(f"Warning: Speaker {speaker_id} directory not found")
                continue
            
            print(f"\nProcessing speaker {speaker_id}...")
            
            for emotion in emotion_categories:
                emotion_dir = speaker_dir / emotion
                if not emotion_dir.exists():
                    print(f"Warning: {emotion_dir} does not exist")
                    continue
                
                # Get all wav files
                audio_files = list(emotion_dir.glob('*.wav'))
                
                # Process each audio file
                for audio_file in tqdm(audio_files[:50], desc=f"{speaker_id}/{emotion}", leave=False):
                    # Extract dimensional emotions
                    emotions = self.extract_emotions(str(audio_file))
                    
                    # Create unified result entry with 4 columns as requested
                    result = {
                        'file_path': str(audio_file),
                        'opensmile_features': None,  # Placeholder for OpenSMILE features (to be filled later)
                        'categorical_label': emotion,  # Original ESD categorical emotion
                        'dimensional_labels': {
                            'arousal': emotions['arousal'],
                            'valence': emotions['valence'],
                            'dominance': emotions['dominance']
                        }
                    }
                    all_results.append(result)
        
        # Create unified DataFrame
        df = pd.DataFrame(all_results)
        
        # Save unified dataset
        output_file = output_path / 'esd_unified_dataset.pkl'
        df.to_pickle(output_file)
        print(f"\nUnified dataset saved to {output_file}")
        
        # Save as CSV for easy inspection (flatten dimensional labels for CSV)
        df_csv = df.copy()
        df_csv['arousal'] = df_csv['dimensional_labels'].apply(lambda x: x['arousal'])
        df_csv['valence'] = df_csv['dimensional_labels'].apply(lambda x: x['valence'])
        df_csv['dominance'] = df_csv['dimensional_labels'].apply(lambda x: x['dominance'])
        df_csv = df_csv.drop('dimensional_labels', axis=1)
        
        csv_file = output_path / 'esd_unified_dataset.csv'
        df_csv.to_csv(csv_file, index=False)
        print(f"CSV version saved to {csv_file}")
        
        # Print dataset info
        self.print_dataset_info(df)
        
        return df
    
    def print_dataset_info(self, df: pd.DataFrame):
        """Print statistics about the unified dataset"""
        print(f"\nUnified Dataset Information:")
        print(f"Total samples: {len(df)}")
        print(f"Categorical emotions: {df['categorical_label'].unique()}")
        print(f"Distribution by categorical emotion:")
        print(df['categorical_label'].value_counts())
        
        # Extract dimensional scores for statistics
        arousal_scores = [x['arousal'] for x in df['dimensional_labels']]
        valence_scores = [x['valence'] for x in df['dimensional_labels']]
        dominance_scores = [x['dominance'] for x in df['dimensional_labels']]
        
        print(f"\nDimensional emotion statistics:")
        print(f"\nArousal:")
        print(f"  Mean: {np.mean(arousal_scores):.3f}")
        print(f"  Std:  {np.std(arousal_scores):.3f}")
        print(f"  Min:  {np.min(arousal_scores):.3f}")
        print(f"  Max:  {np.max(arousal_scores):.3f}")
        
        print(f"\nValence:")
        print(f"  Mean: {np.mean(valence_scores):.3f}")
        print(f"  Std:  {np.std(valence_scores):.3f}")
        print(f"  Min:  {np.min(valence_scores):.3f}")
        print(f"  Max:  {np.max(valence_scores):.3f}")
        
        print(f"\nDominance:")
        print(f"  Mean: {np.mean(dominance_scores):.3f}")
        print(f"  Std:  {np.std(dominance_scores):.3f}")
        print(f"  Min:  {np.min(dominance_scores):.3f}")
        print(f"  Max:  {np.max(dominance_scores):.3f}")
        
        # Show mean dimensional scores by categorical emotion
        print(f"\nMean dimensional scores by categorical emotion:")
        for emotion in df['categorical_label'].unique():
            emotion_mask = df['categorical_label'] == emotion
            emotion_data = df[emotion_mask]['dimensional_labels']
            
            emotion_arousal = np.mean([x['arousal'] for x in emotion_data])
            emotion_valence = np.mean([x['valence'] for x in emotion_data])
            emotion_dominance = np.mean([x['dominance'] for x in emotion_data])
            
            print(f"  {emotion}: A={emotion_arousal:.3f}, V={emotion_valence:.3f}, D={emotion_dominance:.3f}")
        
        print(f"\nDataFrame columns: {list(df.columns)}")
        print(f"OpenSMILE features column (placeholder): {df['opensmile_features'].isna().sum()} null values")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract dimensional emotions using w2v2 model')
    parser.add_argument('--esd_data_dir', type=str,
                       default='/Users/k.sridhara.murthy/Documents/kusha_research/data/ESD',
                       help='Path to ESD dataset')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/k.sridhara.murthy/Documents/kusha_research/paper1/data',
                       help='Output directory for results')
    parser.add_argument('--cache_dir', type=str,
                       default='/Users/k.sridhara.murthy/Documents/kusha_research/paper1/cache',
                       help='Cache directory for model download')
    parser.add_argument('--model_dir', type=str,
                       default='/Users/k.sridhara.murthy/Documents/kusha_research/paper1/models',
                       help='Model directory')
    
    args = parser.parse_args()
    
    # Create emotion extractor
    print("Initializing W2V2 emotion extractor...")
    extractor = W2V2EmotionExtractor(
        cache_dir=args.cache_dir,
        model_dir=args.model_dir
    )
    
    # Download and load model
    print("Loading emotion recognition model...")
    if not extractor.download_and_load_model():
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Process ESD dataset
    print("Processing ESD dataset...")
    df = extractor.process_esd_dataset(args.esd_data_dir, args.output_dir)
    
    print("\nDimensional emotion extraction complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()