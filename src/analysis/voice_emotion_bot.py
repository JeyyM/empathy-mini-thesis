import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import warnings
import cv2
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import os

# Wav2vec2 emotion model imports
try:
    import audonnx
    AUDONNX_AVAILABLE = True
except ImportError:
    AUDONNX_AVAILABLE = False
    print("Warning: audonnx not available. Install with: pip install audonnx")
    print("Falling back to rule-based emotion detection.")

# Suppress warnings
warnings.filterwarnings("ignore")

class VoiceEmotionBot:
    def __init__(self, sample_rate=22050, use_ml_model=True, model_path='emotion_model'):
        """
        Hybrid Voice Emotion Detection Bot using pre-trained Wav2vec2 model + acoustic features
        
        IMPROVED APPROACH:
        - Primary: Wav2vec2 transformer model (93%+ accuracy) for arousal/valence/dominance
        - Secondary: Keep all acoustic features for analysis and compatibility
        - Fallback: Rule-based approach if ML model unavailable
        
        Features extracted:
        - Prosodic: pitch, speaking rate, volume, voice tremor
        - Spectral: MFCCs, spectral features, voice quality
        - Temporal: pause patterns, rhythm, breath patterns
        - ML-based dimensions: arousal, valence, dominance (from Wav2vec2)
        - Derived: intensity, stress, basic emotions
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_length = 2048
        
        # Initialize feature scaler for normalization
        self.scaler = StandardScaler()
        
        # Try to load Wav2vec2 emotion model
        self.ml_model = None
        self.use_ml_model = use_ml_model and AUDONNX_AVAILABLE
        
        if self.use_ml_model:
            try:
                if os.path.exists(model_path):
                    print(f"Loading Wav2vec2 emotion model from: {model_path}")
                    self.ml_model = audonnx.load(model_path)
                    print("✓ ML model loaded successfully!")
                    print("  Using transformer-based arousal/valence/dominance prediction")
                else:
                    print(f"Warning: Model path not found: {model_path}")
                    print("  Run download_emotion_model.py to download the model")
                    print("  Falling back to rule-based approach")
                    self.use_ml_model = False
            except Exception as e:
                print(f"Warning: Could not load ML model: {e}")
                print("  Falling back to rule-based approach")
                self.use_ml_model = False
        
        if not self.use_ml_model:
            print("Using rule-based voice emotion detection")
        
        # Voice emotion-to-dimension mappings (for basic emotions)
        self.voice_emotion_dimensions = {
            'angry': {'arousal': 0.8, 'valence': -0.8, 'intensity': 0.9, 'stress': 0.9},
            'happy': {'arousal': 0.6, 'valence': 0.8, 'intensity': 0.7, 'stress': 0.2},
            'sad': {'arousal': -0.7, 'valence': -0.6, 'intensity': 0.6, 'stress': 0.3},
            'fear': {'arousal': 0.9, 'valence': -0.7, 'intensity': 0.8, 'stress': 0.9},
            'surprise': {'arousal': 0.8, 'valence': 0.2, 'intensity': 0.7, 'stress': 0.4},
            'disgust': {'arousal': 0.3, 'valence': -0.8, 'intensity': 0.6, 'stress': 0.6},
            'neutral': {'arousal': 0.0, 'valence': 0.1, 'intensity': 0.2, 'stress': 0.1}
        }
        
        # Initialize data storage
        self.voice_data = []
        self.last_analysis_time = None
        
        print("Voice Emotion Bot initialized")
        print("Features: Pitch, MFCCs, Spectral, Prosodic, ML Emotional dimensions")
    
    def extract_voice_features(self, audio_segment, sr):
        """
        Extract comprehensive voice features from audio segment
        
        Returns:
        - Prosodic features: pitch, speaking rate, volume
        - Spectral features: MFCCs, spectral characteristics  
        - Voice quality: harmonic-to-noise ratio, voice tremor
        - Emotional dimensions: arousal, valence, intensity, stress
        """
        features = {}
        
        try:
            # 1. PITCH FEATURES (Fundamental Frequency)
            pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr, 
                                                  hop_length=self.hop_length,
                                                  fmin=80, fmax=400)
            
            # Extract pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0 else 0
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
                features['pitch_variation'] = features['pitch_std'] / (features['pitch_mean'] + 1e-6)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
                features['pitch_variation'] = 0
            
            # 2. ENERGY/VOLUME FEATURES
            rms_energy = librosa.feature.rms(y=audio_segment, hop_length=self.hop_length)[0]
            features['volume_mean'] = np.mean(rms_energy)
            features['volume_std'] = np.std(rms_energy)
            features['volume_range'] = np.max(rms_energy) - np.min(rms_energy)
            
            # 3. SPECTRAL FEATURES
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
            features['spectral_centroid'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr)[0]
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)[0]
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            # Zero crossing rate (indicator of speech vs noise)
            zcr = librosa.feature.zero_crossing_rate(audio_segment, hop_length=self.hop_length)[0]
            features['zero_crossing_rate'] = np.mean(zcr)
            
            # 4. MFCC FEATURES (Mel-Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
            # 5. TEMPORAL FEATURES
            # Speech rate estimation (based on onset detection)
            onset_frames = librosa.onset.onset_detect(y=audio_segment, sr=sr)
            features['speech_rate'] = len(onset_frames) / (len(audio_segment) / sr)  # onsets per second
            
            # Silence/pause analysis
            silence_threshold = 0.01
            silence_frames = rms_energy < silence_threshold
            features['silence_ratio'] = np.sum(silence_frames) / len(silence_frames)
            
            # 6. VOICE QUALITY FEATURES
            # Harmonic-to-noise ratio estimation
            harmonic = librosa.effects.harmonic(audio_segment)
            percussive = librosa.effects.percussive(audio_segment)
            harmonic_energy = np.sum(harmonic**2)
            percussive_energy = np.sum(percussive**2)
            features['harmonic_ratio'] = harmonic_energy / (harmonic_energy + percussive_energy + 1e-6)
            
            # Voice tremor (pitch instability)
            if len(pitch_values) > 1:
                pitch_diff = np.diff(pitch_values)
                features['voice_tremor'] = np.std(pitch_diff)
            else:
                features['voice_tremor'] = 0
            
        except Exception as e:
            print(f"Warning: Error extracting features: {e}")
            # Return default features if extraction fails
            return self._get_default_features()
        
        return features
    
    def _get_default_features(self):
        """Return default features when extraction fails"""
        features = {}
        feature_names = ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_variation',
                        'volume_mean', 'volume_std', 'volume_range',
                        'spectral_centroid', 'spectral_centroid_std', 'spectral_bandwidth',
                        'spectral_rolloff', 'zero_crossing_rate', 'speech_rate',
                        'silence_ratio', 'harmonic_ratio', 'voice_tremor']
        
        for name in feature_names:
            features[name] = 0.0
        
        # Add MFCC features
        for i in range(13):
            features[f'mfcc_{i+1}'] = 0.0
            features[f'mfcc_{i+1}_std'] = 0.0
            
        return features
    
    def features_to_emotions(self, features, audio_segment=None, sr=None):
        """
        Convert voice features to emotional dimensions and basic emotions
        
        HYBRID APPROACH:
        1. If ML model available: Use Wav2vec2 for arousal/valence/dominance (primary)
        2. Keep acoustic features for intensity/stress calculation
        3. Derive basic emotions from ML dimensions + acoustic features
        4. Fallback to rule-based if ML unavailable
        """
        emotions = {}
        
        # Try to get ML-based emotions first
        ml_emotions = None
        if self.use_ml_model and self.ml_model is not None and audio_segment is not None and sr is not None:
            ml_emotions = self._get_ml_emotions(audio_segment, sr)
        
        # Calculate acoustic-based normalized features (always needed)
        pitch_norm = min(1.0, features['pitch_mean'] / 300.0)
        volume_norm = min(1.0, features['volume_mean'] * 25)
        pitch_var_norm = min(1.0, features['pitch_variation'])
        speech_rate_norm = min(1.0, features['speech_rate'] / 5.0)
        
        # === PRIMARY: Use ML model if available ===
        if ml_emotions is not None:
            # Use ML predictions for core dimensions (arousal, valence, dominance)
            emotions['voice_arousal'] = ml_emotions['arousal']
            emotions['voice_valence'] = ml_emotions['valence']
            emotions['voice_dominance'] = ml_emotions.get('dominance', 0.0)
            
            # Calculate intensity from acoustic features (ML doesn't predict this)
            intensity = (0.4 * volume_norm + 0.3 * pitch_var_norm + 0.3 * (1 - features['silence_ratio']))
            emotions['voice_intensity'] = intensity
            
            # Calculate stress from acoustic features + ML arousal
            stress = (0.25 * pitch_norm + 0.35 * pitch_var_norm + 
                     0.2 * (1 - features['harmonic_ratio']) + 0.2 * abs(ml_emotions['arousal']))
            emotions['voice_stress'] = min(1.0, stress * 1.2)
            
        # === FALLBACK: Rule-based approach ===
        else:
            # AROUSAL: High pitch variation + high speech rate + high volume = high arousal
            arousal = (0.5 * pitch_var_norm + 0.35 * speech_rate_norm + 0.15 * volume_norm)
            emotions['voice_arousal'] = np.clip((arousal * 2.8) - 1.0, -1, 1)
            
            # VALENCE: Spectral characteristics
            spectral_norm = min(1.0, features['spectral_centroid'] / 5000.0)
            pitch_harshness = min(1.0, features['pitch_variation'] / 100.0)
            
            valence_raw = (0.3 * spectral_norm + 0.15 * (1 - features['silence_ratio']) + 
                          0.2 * features['harmonic_ratio'] - 0.35 * pitch_harshness)
            emotions['voice_valence'] = np.clip((valence_raw * 2.2) - 1, -1, 1)
            
            # INTENSITY: Overall voice energy and variation
            intensity = (0.4 * volume_norm + 0.3 * pitch_var_norm + 0.3 * (1 - features['silence_ratio']))
            emotions['voice_intensity'] = intensity
            
            # STRESS: High pitch + high variation + low harmonic ratio = stress
            stress = (0.3 * pitch_norm + 0.4 * pitch_var_norm + 0.3 * (1 - features['harmonic_ratio']))
            emotions['voice_stress'] = min(1.0, stress * 1.4)
            
            # Dominance (not in original, but useful)
            emotions['voice_dominance'] = np.clip(volume_norm * 2 - 1, -1, 1)
        
        # === BASIC EMOTIONS (derived from dimensions + acoustic features) ===
        # Use arousal/valence from whichever source (ML or rule-based)
        arousal = (emotions['voice_arousal'] + 1) / 2  # Convert to 0-1
        valence = (emotions['voice_valence'] + 1) / 2  # Convert to 0-1
        
        # BALANCED FORMULAS - Calibrated to prevent angry over-detection
        
        # Happy: High valence + moderate-high arousal + harmonic voice
        happy_score = (valence ** 1.2) * (arousal + 0.2) * features['harmonic_ratio']
        emotions['voice_happy'] = min(1.0, happy_score * 1.6)
        
        # Angry: Negative valence (PRIMARY) + acoustic harshness
        # Works for both high-arousal (shouting) and low-arousal (cold) anger
        # STRICT gate: only detect if CLEARLY negative valence
        if valence < 0.40:  # Only if strongly negative (< -0.2 in -1 to 1 scale)
            angry_intensity = max(volume_norm, pitch_var_norm * 0.7)
            # Require BOTH negative valence AND acoustic intensity
            # Lower multiplier to reduce false positives
            angry_score = ((1 - valence) ** 2.0) * ((arousal + 0.5) ** 0.6) * (angry_intensity + 0.2) * (pitch_var_norm + 0.25)
            emotions['voice_angry'] = min(1.0, angry_score * 2.0)
        else:
            emotions['voice_angry'] = 0.0
        
        # Sad: Low arousal + negative valence + low volume + silence
        sad_score = ((1 - arousal) ** 1.3) * ((1 - valence) ** 1.2) * (1 - volume_norm + 0.2) * (features['silence_ratio'] + 0.2)
        emotions['voice_sad'] = min(1.0, sad_score * 2.2)
        
        
        # Fear: High arousal + negative valence + tremor + stress
        fear_score = (arousal ** 1.1) * ((1 - valence) ** 1.1) * min(1.0, features['voice_tremor'] / 35.0) * emotions['voice_stress']
        emotions['voice_fear'] = min(1.0, fear_score * 1.6)
        
        # Surprise: Very high arousal + neutral valence + high pitch variation
        surprise_score = (arousal ** 1.5) * (1 - abs(emotions['voice_valence'])) * pitch_var_norm
        emotions['voice_surprise'] = min(1.0, surprise_score * 1.2)
        
        # Disgust: Moderate arousal + negative valence + low harmonic ratio
        disgust_score = (arousal ** 0.8) * ((1 - valence) ** 1.2) * (1 - features['harmonic_ratio'])
        emotions['voice_disgust'] = min(1.0, disgust_score * 1.5)
        
        # Neutral: Low arousal + neutral valence + stable harmonic features
        neutral_score = ((1 - arousal) ** 1.1) * (1 - abs(emotions['voice_valence'])) * features['harmonic_ratio']
        emotions['voice_neutral'] = min(1.0, neutral_score * 1.8)
        
        # Normalize basic emotions to sum to 1
        emotion_sum = sum([emotions[key] for key in emotions if key.startswith('voice_') and 
                          key not in ['voice_arousal', 'voice_valence', 'voice_intensity', 'voice_stress', 'voice_dominance']])
        
        if emotion_sum > 0:
            for key in emotions:
                if key.startswith('voice_') and key not in ['voice_arousal', 'voice_valence', 'voice_intensity', 'voice_stress', 'voice_dominance']:
                    emotions[key] = emotions[key] / emotion_sum
        
        return emotions
    
    def _get_ml_emotions(self, audio_segment, sr):
        """
        Get emotion predictions from Wav2vec2 ML model
        
        Returns:
        - arousal: -1 to 1 (calm to exciting)
        - valence: -1 to 1 (negative to positive)  
        - dominance: -1 to 1 (submissive to dominant)
        """
        try:
            # Resample to 16kHz if needed (Wav2vec2 requirement)
            if sr != 16000:
                audio_16k = librosa.resample(audio_segment, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio_segment
            
            # Ensure float32 format
            audio_16k = audio_16k.astype(np.float32)
            
            # Get predictions from model
            # Model outputs: {'logits': array([arousal, dominance, valence]), 'hidden_states': ...}
            output = self.ml_model(audio_16k, 16000)
            logits = output['logits'][0]  # Shape: (3,) containing [arousal, dominance, valence]
            
            # Convert from 0-1 scale to -1 to 1 scale
            arousal = (logits[0] * 2) - 1
            dominance = (logits[1] * 2) - 1
            valence = (logits[2] * 2) - 1
            
            return {
                'arousal': float(np.clip(arousal, -1, 1)),
                'valence': float(np.clip(valence, -1, 1)),
                'dominance': float(np.clip(dominance, -1, 1))
            }
            
        except Exception as e:
            print(f"Warning: ML emotion prediction failed: {e}")
            return None
    
    def process_audio_file(self, audio_path, segment_duration=1.0):
        """
        Process audio file and extract voice emotions over time
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each analysis segment in seconds
        
        Returns:
            DataFrame with voice emotion data over time
        """
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return pd.DataFrame()
        
        print(f"Loading audio file: {audio_path}")
        
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / sr
            
            print(f"Audio loaded: {duration:.2f} seconds, Sample rate: {sr}")
            
            # Process audio in segments
            segment_samples = int(segment_duration * sr)
            voice_data = []
            
            for start_sample in range(0, len(audio), segment_samples):
                end_sample = min(start_sample + segment_samples, len(audio))
                segment = audio[start_sample:end_sample]
                
                # Skip very short segments
                if len(segment) < sr * 0.1:  # Skip segments shorter than 0.1 seconds
                    continue
                
                # Calculate timestamp
                time_seconds = start_sample / sr
                timestamp = datetime.now() + timedelta(seconds=time_seconds)
                
                print(f"Processing segment at {time_seconds:.1f}s...")
                
                # Extract features
                features = self.extract_voice_features(segment, sr)
                
                # Convert to emotions (pass audio segment for ML model)
                emotions = self.features_to_emotions(features, audio_segment=segment, sr=sr)
                
                # Create data entry
                entry = {
                    'timestamp': timestamp,
                    'time_seconds': time_seconds,
                    'segment_start': start_sample,
                    'segment_end': end_sample,
                    **features,
                    **emotions
                }
                
                voice_data.append(entry)
            
            # Convert to DataFrame
            df = pd.DataFrame(voice_data)
            self.voice_data = voice_data
            
            print(f"Processed {len(df)} voice segments")
            return df
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return pd.DataFrame()
    
    def get_voice_summary(self):
        """Get summary statistics of voice emotions"""
        if not self.voice_data:
            return {}
        
        df = pd.DataFrame(self.voice_data)
        
        # Calculate summary for each emotion dimension
        emotion_cols = ['voice_arousal', 'voice_valence', 'voice_intensity', 'voice_stress',
                       'voice_angry', 'voice_disgust', 'voice_fear', 'voice_happy', 
                       'voice_sad', 'voice_surprise', 'voice_neutral']
        
        summary = {}
        for emotion in emotion_cols:
            if emotion in df.columns:
                summary[emotion] = {
                    'mean': df[emotion].mean(),
                    'std': df[emotion].std(),
                    'min': df[emotion].min(),
                    'max': df[emotion].max()
                }
        
        return summary
    
    def plot_voice_emotions(self, save_path=None):
        """Plot voice emotions over time"""
        if not self.voice_data:
            print("No voice data to plot")
            return
        
        df = pd.DataFrame(self.voice_data)
        
        # Create subplot for different emotion aspects
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Emotional dimensions
        axes[0].plot(df['time_seconds'], df['voice_arousal'], label='Arousal', linewidth=2)
        axes[0].plot(df['time_seconds'], df['voice_valence'], label='Valence', linewidth=2)
        axes[0].plot(df['time_seconds'], df['voice_intensity'], label='Intensity', linewidth=2)
        axes[0].plot(df['time_seconds'], df['voice_stress'], label='Stress', linewidth=2)
        axes[0].set_title('Voice Emotional Dimensions Over Time')
        axes[0].set_ylabel('Emotion Level')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-1.1, 1.1)
        
        # Plot 2: Basic emotions
        emotion_cols = ['voice_angry', 'voice_happy', 'voice_sad', 'voice_fear', 
                       'voice_surprise', 'voice_disgust', 'voice_neutral']
        for emotion in emotion_cols:
            if emotion in df.columns:
                axes[1].plot(df['time_seconds'], df[emotion], 
                           label=emotion.replace('voice_', '').title(), linewidth=2)
        
        axes[1].set_title('Voice Basic Emotions Over Time')
        axes[1].set_ylabel('Emotion Probability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # Plot 3: Voice features
        axes[2].plot(df['time_seconds'], df['pitch_mean'], label='Pitch (Hz)', linewidth=2)
        axes[2].plot(df['time_seconds'], df['volume_mean'] * 1000, label='Volume (x1000)', linewidth=2)
        axes[2].plot(df['time_seconds'], df['speech_rate'] * 50, label='Speech Rate (x50)', linewidth=2)
        axes[2].set_title('Voice Features Over Time')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Feature Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Voice emotion plot saved to: {save_path}")
        
        plt.show()
    
    def plot_voice_heatmap(self, save_path=None, bins=20):
        """
        Plot voice emotion distribution heatmaps similar to facial heatmap
        Shows arousal-valence space, intensity distribution, and emotion patterns
        """
        if not self.voice_data:
            print("No voice data to plot")
            return
        
        df = pd.DataFrame(self.voice_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Voice Emotion Distribution Heatmaps', fontsize=16, fontweight='bold')
        
        # 1. Arousal-Valence density heatmap
        ax1 = axes[0, 0]
        arousal_vals = df['voice_arousal'].values
        valence_vals = df['voice_valence'].values
        
        h1 = ax1.hist2d(valence_vals, arousal_vals, bins=bins, cmap='YlOrRd', alpha=0.8)
        ax1.set_xlabel('Valence (Negative ← → Positive)')
        ax1.set_ylabel('Arousal (Calm ← → Exciting)')
        ax1.set_title('Voice Arousal-Valence Density')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax1.text(0.6, 0.6, 'Excited\n(Happy)', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(-0.6, 0.6, 'Stressed\n(Angry/Fear)', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(0.6, -0.6, 'Calm\n(Neutral)', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(-0.6, -0.6, 'Tired\n(Sad)', ha='center', va='center', fontsize=10, fontweight='bold')
        
        plt.colorbar(h1[3], ax=ax1, label='Frequency')
        
        # 2. Intensity heatmap
        ax2 = axes[0, 1]
        intensity_vals = df['voice_intensity'].values
        h2 = ax2.hist2d(valence_vals, arousal_vals, bins=bins, weights=intensity_vals, cmap='plasma', alpha=0.8)
        ax2.set_xlabel('Valence (Negative ← → Positive)')
        ax2.set_ylabel('Arousal (Calm ← → Exciting)')
        ax2.set_title('Voice Emotional Intensity Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='white', linestyle='--', alpha=0.7)
        ax2.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        plt.colorbar(h2[3], ax=ax2, label='Average Intensity')
        
        # 3. Stress level heatmap
        ax3 = axes[1, 0]
        stress_vals = df['voice_stress'].values
        h3 = ax3.hist2d(valence_vals, arousal_vals, bins=bins, weights=stress_vals, cmap='RdYlGn_r', alpha=0.8)
        ax3.set_xlabel('Valence (Negative ← → Positive)')
        ax3.set_ylabel('Arousal (Calm ← → Exciting)')
        ax3.set_title('Voice Stress Level Distribution')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='white', linestyle='--', alpha=0.7)
        ax3.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        plt.colorbar(h3[3], ax=ax3, label='Average Stress')
        
        # 4. Time evolution heatmap (matching facial heatmap)
        ax4 = axes[1, 1]
        if 'time_seconds' in df.columns:
            time_vals = df['time_seconds'].values
            h4 = ax4.hist2d(time_vals, arousal_vals, bins=bins, cmap='viridis', alpha=0.8)
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Arousal (Calm ← → Exciting)')
            ax4.set_title('Voice Arousal Evolution Over Time')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='white', linestyle='--', alpha=0.7)
            plt.colorbar(h4[3], ax=ax4, label='Frequency')
        else:
            ax4.text(0.5, 0.5, 'No time data available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Time Evolution (No Data)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Voice heatmap saved to: {save_path}")
        
        plt.show()
    
    def save_voice_data(self, filename):
        """Save voice emotion data to CSV"""
        if not self.voice_data:
            print("No voice data to save")
            return
        
        df = pd.DataFrame(self.voice_data)
        df.to_csv(filename, index=False)
        print(f"Voice emotion data saved to: {filename}")
    
    def plot_voice_movement_heatmap(self, save_path=None, grid_size=50):
        """Create a heatmap showing voice emotion movement patterns within the emotion circle"""
        if not self.voice_data:
            print("No voice data available for movement heatmap")
            return
        
        df = pd.DataFrame(self.voice_data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Voice Emotion Circle Movement Analysis', fontsize=16, fontweight='bold')
        
        # Extract arousal and valence data
        arousal_vals = df['voice_arousal'].values
        valence_vals = df['voice_valence'].values
        
        # Create circular grid for heatmap
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Create mask for circular boundary
        circle_mask = X**2 + Y**2 <= 1
        
        # 1. Movement density heatmap (left plot)
        H, xedges, yedges = np.histogram2d(valence_vals, arousal_vals, bins=grid_size, range=[[-1, 1], [-1, 1]])
        H = H.T  # Transpose for correct orientation
        
        # Apply circular mask
        H_masked = np.where(circle_mask, H, np.nan)
        
        im1 = ax1.imshow(H_masked, extent=[-1, 1, -1, 1], origin='lower', cmap='YlOrRd', alpha=0.8)
        
        # Draw circle boundary
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax1.add_patch(circle)
        
        # Draw quadrant lines
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Add quadrant labels
        ax1.text(0.7, 0.7, 'EXCITED\nEnergized', ha='center', va='center', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        ax1.text(-0.7, 0.7, 'STRESSED\nAnxious', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
        ax1.text(0.7, -0.7, 'CALM\nRelaxed', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax1.text(-0.7, -0.7, 'TIRED\nLow mood', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Add axis labels
        ax1.set_xlabel('MOOD: Negative ← → Positive', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ENERGY: Low ← → High', fontsize=12, fontweight='bold')
        ax1.set_title('Where Voice Spends Time\n(Movement Density)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Time Spent (frequency)', fontsize=10)
        
        # Set equal aspect ratio and limits
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_aspect('equal')
        
        # 2. Movement path with intensity (right plot)
        # Create a smooth heatmap based on trajectory
        gaussian_grid = np.zeros((grid_size, grid_size))
        
        for i, (val, ar) in enumerate(zip(valence_vals, arousal_vals)):
            # Convert to grid coordinates
            x_idx = int((val + 1) * grid_size / 2)
            y_idx = int((ar + 1) * grid_size / 2)
            
            # Ensure indices are within bounds
            x_idx = max(0, min(grid_size - 1, x_idx))
            y_idx = max(0, min(grid_size - 1, y_idx))
            
            # Add Gaussian blob around the point
            sigma = 3  # Spread of the Gaussian
            for dx in range(-sigma*2, sigma*2 + 1):
                for dy in range(-sigma*2, sigma*2 + 1):
                    nx, ny = x_idx + dx, y_idx + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        # Calculate Gaussian weight
                        dist = np.sqrt(dx**2 + dy**2)
                        weight = np.exp(-(dist**2) / (2 * sigma**2))
                        gaussian_grid[ny, nx] += weight
        
        # Apply circular mask
        gaussian_grid_masked = np.where(circle_mask, gaussian_grid, np.nan)
        
        im2 = ax2.imshow(gaussian_grid_masked, extent=[-1, 1, -1, 1], origin='lower', cmap='plasma', alpha=0.8)
        
        # Draw circle boundary
        circle2 = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax2.add_patch(circle2)
        
        # Draw quadrant lines
        ax2.axhline(y=0, color='white', linestyle='-', alpha=0.5, linewidth=1)
        ax2.axvline(x=0, color='white', linestyle='-', alpha=0.5, linewidth=1)
        
        # Plot trajectory line
        ax2.plot(valence_vals, arousal_vals, 'w-', alpha=0.3, linewidth=1)
        ax2.plot(valence_vals[0], arousal_vals[0], 'go', markersize=10, label='Start', markeredgecolor='white')
        ax2.plot(valence_vals[-1], arousal_vals[-1], 'ro', markersize=10, label='End', markeredgecolor='white')
        
        # Add quadrant labels
        ax2.text(0.7, 0.7, 'EXCITED\nEnergized', ha='center', va='center', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        ax2.text(-0.7, 0.7, 'STRESSED\nAnxious', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
        ax2.text(0.7, -0.7, 'CALM\nRelaxed', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax2.text(-0.7, -0.7, 'TIRED\nLow mood', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Add axis labels
        ax2.set_xlabel('MOOD: Negative ← → Positive', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ENERGY: Low ← → High', fontsize=12, fontweight='bold')
        ax2.set_title('Voice Emotional Journey\n(Movement Path)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Path Intensity', fontsize=10)
        
        # Set equal aspect ratio and limits
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_aspect('equal')
        ax2.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Voice movement heatmap saved to: {save_path}")
        
        plt.show()

def main():
    """Example usage of Voice Emotion Bot"""
    # Create voice emotion bot
    bot = VoiceEmotionBot()
    
    # Example audio file (you'll need to provide this)
    audio_path = "test_audio.wav"  # Replace with your audio file
    
    if os.path.exists(audio_path):
        # Process audio file
        df = bot.process_audio_file(audio_path, segment_duration=1.0)
        
        if not df.empty:
            print(f"\nProcessed {len(df)} voice emotion segments")
            
            # Show summary
            summary = bot.get_voice_summary()
            print("\nVoice Emotion Summary:")
            for emotion, stats in summary.items():
                print(f"{emotion}: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}")
            
            # Find dominant voice emotion
            emotion_cols = ['voice_angry', 'voice_happy', 'voice_sad', 'voice_fear', 
                           'voice_surprise', 'voice_disgust', 'voice_neutral']
            means = {col.replace('voice_', ''): df[col].mean() for col in emotion_cols if col in df.columns}
            dominant = max(means, key=means.get)
            print(f"\nDominant voice emotion: {dominant.upper()} ({means[dominant]:.3f})")
            
            # Plot emotions
            bot.plot_voice_emotions("voice_emotions.png")
            
            # Save data
            bot.save_voice_data("voice_emotion_data.csv")
        else:
            print("No voice emotions detected")
    else:
        print(f"Audio file not found: {audio_path}")
        print("Please provide an audio file to analyze")

if __name__ == "__main__":
    main()