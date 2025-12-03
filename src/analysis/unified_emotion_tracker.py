from src.analysis.emotion_bot import EmotionBot
from src.analysis.voice_emotion_bot import VoiceEmotionBot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import librosa
import moviepy.editor as mp
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

#####

class UnifiedEmotionTracker:
    """
    Unified tracker that synchronizes facial and voice emotion analysis
    Processes video files to extract both visual and audio emotion data
    """
    
    def __init__(self, sample_rate=22050, use_ml_voice=False):
        print("Initializing Unified Emotion Tracker...")
        
        # Initialize both emotion bots
        self.face_bot = EmotionBot()
        self.voice_bot = VoiceEmotionBot(sample_rate=sample_rate, use_ml_model=use_ml_voice)
        
        self.sample_rate = sample_rate
        self.synchronized_data = []
        
        print("[OK] Facial emotion detector ready")
        if use_ml_voice:
            print("[OK] Voice emotion detector ready (ML-enhanced with Wav2vec2)")
        else:
            print("[OK] Voice emotion detector ready (rule-based)")
        print("[OK] Unified tracker initialized")
    
    def extract_audio_from_video(self, video_path, temp_audio_path="temp_audio.wav"):
        """Extract audio from video file for voice analysis"""
        try:
            print(f"Extracting audio from video: {video_path}")
            
            # Load video with moviepy
            video_clip = mp.VideoFileClip(video_path)
            
            # Extract audio
            audio_clip = video_clip.audio
            
            if audio_clip is None:
                print("⚠️  No audio track found in video")
                return None, None
            
            # Write audio to temporary file
            audio_clip.write_audiofile(temp_audio_path, 
                                     fps=self.sample_rate, 
                                     verbose=False, 
                                     logger=None)
            
            # Get video duration for synchronization
            video_duration = video_clip.duration
            
            # Clean up
            audio_clip.close()
            video_clip.close()
            
            print(f"[OK] Audio extracted: {video_duration:.2f} seconds")
            return temp_audio_path, video_duration
            
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            return None, None
    
    def process_video_unified(self, video_path, sample_interval=1.0, cleanup_temp=True):
        """
        Process video file with synchronized facial and voice emotion analysis
        
        Args:
            video_path: Path to video file
            sample_interval: Time interval between samples (seconds)
            cleanup_temp: Whether to delete temporary audio file
        
        Returns:
            DataFrame with synchronized facial and voice emotion data
        """
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return pd.DataFrame()
        
        print(f"[VIDEO] Processing video: {video_path}")
        print(f"[INFO] Sample interval: {sample_interval} seconds")
        
        # Step 1: Extract audio from video
        temp_audio_path, video_duration = self.extract_audio_from_video(video_path)
        
        # Step 2: Process facial emotions
        print("\n[FACIAL] Processing facial emotions...")
        facial_df = self.face_bot.process_video(video_path, sample_rate=sample_interval)
        
        # Step 3: Process voice emotions (if audio available)
        voice_df = pd.DataFrame()
        if temp_audio_path and os.path.exists(temp_audio_path):
            print("\n[VOICE] Processing voice emotions...")
            voice_df = self.voice_bot.process_audio_file(temp_audio_path, segment_duration=sample_interval)
            
            # Clean up temporary audio file
            if cleanup_temp:
                try:
                    os.remove(temp_audio_path)
                    print("[CLEANUP] Cleaned up temporary audio file")
                except:
                    pass
        else:
            print("[WARNING] No audio data available for voice analysis")
        
        # Step 4: Synchronize the data
        print("\n[SYNC] Synchronizing facial and voice data...")
        synchronized_df = self.synchronize_emotion_data(facial_df, voice_df, sample_interval)
        
        print(f"[OK] Processing complete: {len(synchronized_df)} synchronized samples")
        
        return synchronized_df
    
    def synchronize_emotion_data(self, facial_df, voice_df, sample_interval):
        """
        Synchronize facial and voice emotion data by timestamp
        """
        synchronized_data = []
        
        if facial_df.empty and voice_df.empty:
            print("❌ No emotion data to synchronize")
            return pd.DataFrame()
        
        # Determine time range
        if not facial_df.empty and not voice_df.empty:
            # Both data available - use overlapping time range
            min_time = max(facial_df['time_seconds'].min(), voice_df['time_seconds'].min())
            max_time = min(facial_df['time_seconds'].max(), voice_df['time_seconds'].max())
        elif not facial_df.empty:
            # Only facial data
            min_time = facial_df['time_seconds'].min()
            max_time = facial_df['time_seconds'].max()
        else:
            # Only voice data
            min_time = voice_df['time_seconds'].min()
            max_time = voice_df['time_seconds'].max()
        
        print(f"Synchronizing data from {min_time:.1f}s to {max_time:.1f}s")
        
        # Create synchronized timeline
        current_time = min_time
        while current_time <= max_time:
            entry = {
                'timestamp': datetime.now() + timedelta(seconds=current_time),
                'time_seconds': current_time
            }
            
            # Find nearest facial emotion data
            if not facial_df.empty:
                facial_idx = (facial_df['time_seconds'] - current_time).abs().idxmin()
                facial_row = facial_df.iloc[facial_idx]
                
                # Add facial emotion data
                facial_columns = ['quadrant', 'arousal', 'valence', 'intensity', 'excitement', 
                                'calmness', 'positivity', 'negativity', 'angry', 'disgust', 
                                'fear', 'happy', 'sad', 'surprise', 'neutral']
                
                for col in facial_columns:
                    if col in facial_row:
                        entry[f'facial_{col}'] = facial_row[col]
            
            # Find nearest voice emotion data
            if not voice_df.empty:
                voice_idx = (voice_df['time_seconds'] - current_time).abs().idxmin()
                voice_row = voice_df.iloc[voice_idx]
                
                # Add ALL voice columns (not just a subset)
                # Skip timestamp/time columns, add everything else with voice_ prefix if not already prefixed
                for col in voice_row.index:
                    if col not in ['timestamp', 'time_seconds']:
                        # If column already starts with voice_, keep it as is
                        if col.startswith('voice_'):
                            entry[col] = voice_row[col]
                        else:
                            # Add voice_ prefix to acoustic features
                            entry[f'voice_{col}'] = voice_row[col]
            
            # Calculate combined metrics
            self.calculate_combined_metrics(entry)
            
            synchronized_data.append(entry)
            current_time += sample_interval
        
        df = pd.DataFrame(synchronized_data)
        self.synchronized_data = synchronized_data
        
        return df
    
    def calculate_combined_metrics(self, entry):
        """
        Calculate combined facial + voice emotion metrics
        """
        # Combined arousal (average if both available)
        facial_arousal = entry.get('facial_arousal', 0)
        voice_arousal = entry.get('voice_arousal', 0)
        
        if 'facial_arousal' in entry and 'voice_arousal' in entry:
            entry['combined_arousal'] = (facial_arousal + voice_arousal) / 2
        elif 'facial_arousal' in entry:
            entry['combined_arousal'] = facial_arousal
        elif 'voice_arousal' in entry:
            entry['combined_arousal'] = voice_arousal
        else:
            entry['combined_arousal'] = 0
        
        # Combined valence
        facial_valence = entry.get('facial_valence', 0)
        voice_valence = entry.get('voice_valence', 0)
        
        if 'facial_valence' in entry and 'voice_valence' in entry:
            entry['combined_valence'] = (facial_valence + voice_valence) / 2
        elif 'facial_valence' in entry:
            entry['combined_valence'] = facial_valence
        elif 'voice_valence' in entry:
            entry['combined_valence'] = voice_valence
        else:
            entry['combined_valence'] = 0
        
        # Combined intensity
        facial_intensity = entry.get('facial_intensity', 0)
        voice_intensity = entry.get('voice_intensity', 0)
        
        if 'facial_intensity' in entry and 'voice_intensity' in entry:
            entry['combined_intensity'] = (facial_intensity + voice_intensity) / 2
        elif 'facial_intensity' in entry:
            entry['combined_intensity'] = facial_intensity
        elif 'voice_intensity' in entry:
            entry['combined_intensity'] = voice_intensity
        else:
            entry['combined_intensity'] = 0
        
        # Determine combined emotional state
        arousal = entry['combined_arousal']
        valence = entry['combined_valence']
        
        # Quadrant mapping
        if arousal > 0 and valence > 0:
            entry['combined_quadrant'] = "Excited"
        elif arousal > 0 and valence < 0:
            entry['combined_quadrant'] = "Stressed"
        elif arousal < 0 and valence > 0:
            entry['combined_quadrant'] = "Calm"
        else:
            entry['combined_quadrant'] = "Tired"
    
    def plot_unified_emotions(self, save_path=None):
        """
        Plot unified facial and voice emotions
        """
        if not self.synchronized_data:
            print("❌ No synchronized data to plot")
            return
        
        df = pd.DataFrame(self.synchronized_data)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        
        # Plot 1: Combined emotional dimensions
        time_data = df['time_seconds']
        
        if 'combined_arousal' in df.columns:
            axes[0].plot(time_data, df['combined_arousal'], label='Combined Arousal', linewidth=3, color='red')
        if 'facial_arousal' in df.columns:
            axes[0].plot(time_data, df['facial_arousal'], label='Facial Arousal', linewidth=2, alpha=0.7, linestyle='--')
        if 'voice_arousal' in df.columns:
            axes[0].plot(time_data, df['voice_arousal'], label='Voice Arousal', linewidth=2, alpha=0.7, linestyle=':')
            
        if 'combined_valence' in df.columns:
            axes[0].plot(time_data, df['combined_valence'], label='Combined Valence', linewidth=3, color='blue')
        if 'facial_valence' in df.columns:
            axes[0].plot(time_data, df['facial_valence'], label='Facial Valence', linewidth=2, alpha=0.7, linestyle='--')
        if 'voice_valence' in df.columns:
            axes[0].plot(time_data, df['voice_valence'], label='Voice Valence', linewidth=2, alpha=0.7, linestyle=':')
        
        axes[0].set_title('🎯 Unified Emotional Dimensions (Facial + Voice)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Emotion Level')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-1.1, 1.1)
        
        # Plot 2: Facial vs Voice Basic Emotions (Happy, Angry, Sad)
        key_emotions = ['happy', 'angry', 'sad']
        for emotion in key_emotions:
            facial_col = f'facial_{emotion}'
            voice_col = f'voice_{emotion}'
            
            if facial_col in df.columns:
                axes[1].plot(time_data, df[facial_col], label=f'Facial {emotion.title()}', linewidth=2)
            if voice_col in df.columns:
                axes[1].plot(time_data, df[voice_col], label=f'Voice {emotion.title()}', linewidth=2, linestyle='--')
        
        axes[1].set_title('😊😠😢 Key Emotions: Facial vs Voice Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Emotion Probability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # Plot 3: Voice Features
        if 'pitch_mean' in df.columns:
            axes[2].plot(time_data, df['pitch_mean'], label='Pitch (Hz)', linewidth=2, color='purple')
        if 'volume_mean' in df.columns:
            axes[2].plot(time_data, df['volume_mean'] * 1000, label='Volume (×1000)', linewidth=2, color='orange')
        if 'speech_rate' in df.columns:
            axes[2].plot(time_data, df['speech_rate'] * 50, label='Speech Rate (×50)', linewidth=2, color='green')
        
        axes[2].set_title('Voice Features Over Time', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Feature Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Combined Intensity and Quadrants
        if 'combined_intensity' in df.columns:
            axes[3].plot(time_data, df['combined_intensity'], label='Combined Intensity', linewidth=3, color='black')
        if 'facial_intensity' in df.columns:
            axes[3].plot(time_data, df['facial_intensity'], label='Facial Intensity', linewidth=2, alpha=0.7, color='red')
        if 'voice_intensity' in df.columns:
            axes[3].plot(time_data, df['voice_intensity'], label='Voice Intensity', linewidth=2, alpha=0.7, color='blue')
        
        axes[3].set_title('Combined Emotional Intensity', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].set_ylabel('Intensity Level')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVE] Unified emotion plot saved to: {save_path}")
        
        plt.show()
    
    def get_unified_summary(self):
        """Get summary of unified emotion analysis"""
        if not self.synchronized_data:
            return {}
        
        df = pd.DataFrame(self.synchronized_data)
        
        summary = {
            'total_samples': len(df),
            'duration_seconds': df['time_seconds'].max() - df['time_seconds'].min() if len(df) > 0 else 0,
        }
        
        # Combined metrics summary
        combined_metrics = ['combined_arousal', 'combined_valence', 'combined_intensity']
        for metric in combined_metrics:
            if metric in df.columns:
                summary[metric] = {
                    'mean': df[metric].mean(),
                    'std': df[metric].std(),
                    'min': df[metric].min(),
                    'max': df[metric].max()
                }
        
        # Quadrant distribution
        if 'combined_quadrant' in df.columns:
            quadrant_counts = df['combined_quadrant'].value_counts()
            summary['quadrant_distribution'] = quadrant_counts.to_dict()
            summary['dominant_quadrant'] = quadrant_counts.idxmax() if len(quadrant_counts) > 0 else "Unknown"
        
        # Emotion correlation between facial and voice
        correlations = {}
        emotions = ['arousal', 'valence', 'happy', 'angry', 'sad']
        for emotion in emotions:
            facial_col = f'facial_{emotion}'
            voice_col = f'voice_{emotion}'
            
            if facial_col in df.columns and voice_col in df.columns:
                correlation = df[facial_col].corr(df[voice_col])
                correlations[emotion] = correlation
        
        summary['facial_voice_correlations'] = correlations
        
        return summary
    
    def save_unified_data(self, filename):
        """Save unified emotion data to CSV"""
        if not self.synchronized_data:
            print("❌ No unified data to save")
            return
        
        df = pd.DataFrame(self.synchronized_data)
        df.to_csv(filename, index=False)
        print(f"💾 Unified emotion data saved to: {filename}")

def main():
    """Example usage of Unified Emotion Tracker"""
    import sys
    
    # Get video file from command line arguments
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "testvid.mp4"  # Default fallback
    
    # Create unified tracker
    tracker = UnifiedEmotionTracker()
    
    if os.path.exists(video_path):
        print(f"\n🎬 Processing video: {video_path}")
        
        # Process video with synchronized facial and voice analysis
        df = tracker.process_video_unified(video_path, sample_interval=1.0)
        
        if not df.empty:
            print(f"\n[OK] Analysis complete: {len(df)} synchronized samples")
            
            # Get summary
            summary = tracker.get_unified_summary()
            print(f"\n[SUMMARY] Analysis Summary:")
            print(f"Duration: {summary['duration_seconds']:.1f} seconds")
            print(f"Total samples: {summary['total_samples']}")
            
            if 'dominant_quadrant' in summary:
                print(f"Dominant emotional state: {summary['dominant_quadrant']}")
            
            if 'facial_voice_correlations' in summary:
                print("\n🔗 Facial-Voice Correlations:")
                for emotion, corr in summary['facial_voice_correlations'].items():
                    print(f"  {emotion}: {corr:.3f}")
            
            # Plot unified emotions
            print("\n📈 Generating unified emotion plot...")
            tracker.plot_unified_emotions(f"{video_path}_unified_emotions.png")
            
            # Save data
            tracker.save_unified_data(f"{video_path}_unified_emotion_data.csv")
            
        else:
            print("❌ No emotion data was extracted")
    else:
        print(f"❌ Video file not found: {video_path}")

if __name__ == "__main__":
    main()