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

# Suppress warnings
warnings.filterwarnings("ignore")

class VoiceEmotionBot:
    def __init__(self, sample_rate=22050):
        """
        Voice Emotion Detection Bot that complements facial emotion tracking
        
        Features extracted:
        - Prosodic: pitch, speaking rate, volume, voice tremor
        - Spectral: MFCCs, spectral features, voice quality
        - Temporal: pause patterns, rhythm, breath patterns
        - Emotional dimensions: arousal, valence, intensity, stress
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_length = 2048
        
        # Initialize feature scaler for normalization
        self.scaler = StandardScaler()
        
        # Voice emotion-to-dimension mappings (matching facial system)
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
        
        print("Voice Emotion Bot initialized with librosa")
        print("Features: Pitch, MFCCs, Spectral, Prosodic, Emotional dimensions")
    
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
    
    def features_to_emotions(self, features):
        """
        Convert voice features to emotional dimensions and basic emotions
        Using rule-based approach calibrated for voice characteristics
        """
        emotions = {}
        
        # Normalize features for emotion mapping
        pitch_norm = min(1.0, features['pitch_mean'] / 300.0)  # Normalize pitch to 0-1
        volume_norm = min(1.0, features['volume_mean'] * 10)   # Normalize volume to 0-1
        pitch_var_norm = min(1.0, features['pitch_variation'])  # Already 0-1 range
        speech_rate_norm = min(1.0, features['speech_rate'] / 5.0)  # Normalize speech rate
        
        # AROUSAL: High pitch variation + high speech rate + high volume = high arousal
        arousal = (0.4 * pitch_var_norm + 0.3 * speech_rate_norm + 0.3 * volume_norm)
        emotions['voice_arousal'] = (arousal * 2) - 1  # Scale to -1 to 1
        
        # VALENCE: Higher spectral centroid + moderate pitch = positive valence
        spectral_norm = min(1.0, features['spectral_centroid'] / 5000.0)
        valence_raw = (0.5 * spectral_norm + 0.3 * (1 - features['silence_ratio']) + 
                      0.2 * features['harmonic_ratio'])
        emotions['voice_valence'] = (valence_raw * 2) - 1  # Scale to -1 to 1
        
        # INTENSITY: Overall voice energy and variation
        intensity = (0.4 * volume_norm + 0.3 * pitch_var_norm + 0.3 * (1 - features['silence_ratio']))
        emotions['voice_intensity'] = intensity
        
        # STRESS: High pitch + high variation + low harmonic ratio = stress
        stress = (0.3 * pitch_norm + 0.4 * pitch_var_norm + 0.3 * (1 - features['harmonic_ratio']))
        emotions['voice_stress'] = stress
        
        # BASIC EMOTIONS (rule-based classification)
        # Happy: High valence + moderate arousal + high harmonic ratio
        emotions['voice_happy'] = max(0, (emotions['voice_valence'] + 1) * 0.5 * 
                                     (arousal + 0.5) * features['harmonic_ratio'])
        
        # Angry: High arousal + negative valence + high volume + high pitch variation
        emotions['voice_angry'] = max(0, arousal * max(0, -emotions['voice_valence']) * volume_norm * pitch_var_norm)
        
        # Sad: Low arousal + negative valence + low volume + high silence ratio
        emotions['voice_sad'] = max(0, (1 - arousal) * max(0, -emotions['voice_valence']) * 
                                   (1 - volume_norm) * features['silence_ratio'])
        
        # Fear: High arousal + negative valence + high voice tremor + high stress
        emotions['voice_fear'] = max(0, arousal * max(0, -emotions['voice_valence']) * 
                                    min(1.0, features['voice_tremor'] / 50.0) * stress)
        
        # Surprise: Very high arousal + neutral valence + high pitch variation
        emotions['voice_surprise'] = max(0, min(1.0, arousal * 1.5) * 
                                        (1 - abs(emotions['voice_valence'])) * pitch_var_norm)
        
        # Disgust: Moderate arousal + negative valence + low harmonic ratio
        emotions['voice_disgust'] = max(0, (arousal * 0.7) * max(0, -emotions['voice_valence']) * 
                                       (1 - features['harmonic_ratio']))
        
        # Neutral: Low arousal + neutral valence + stable features
        emotions['voice_neutral'] = max(0, (1 - arousal) * (1 - abs(emotions['voice_valence'])) * 
                                       features['harmonic_ratio'])
        
        # Normalize basic emotions to sum to 1
        emotion_sum = sum([emotions[key] for key in emotions if key.startswith('voice_') and 
                          key not in ['voice_arousal', 'voice_valence', 'voice_intensity', 'voice_stress']])
        
        if emotion_sum > 0:
            for key in emotions:
                if key.startswith('voice_') and key not in ['voice_arousal', 'voice_valence', 'voice_intensity', 'voice_stress']:
                    emotions[key] = emotions[key] / emotion_sum
        
        return emotions
    
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
                
                # Convert to emotions
                emotions = self.features_to_emotions(features)
                
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
    
    def save_voice_data(self, filename):
        """Save voice emotion data to CSV"""
        if not self.voice_data:
            print("No voice data to save")
            return
        
        df = pd.DataFrame(self.voice_data)
        df.to_csv(filename, index=False)
        print(f"Voice emotion data saved to: {filename}")

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