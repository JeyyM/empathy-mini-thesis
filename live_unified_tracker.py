"""
Live Unified Emotion Tracker - Captures facial + voice emotions in real-time
Combines webcam video with microphone audio for simultaneous analysis
"""
import cv2
import pyaudio
import numpy as np
import pandas as pd
import wave
import threading
import time
from datetime import datetime
from emotion_bot import EmotionBot
from voice_emotion_bot import VoiceEmotionBot
import warnings
warnings.filterwarnings("ignore")

class LiveUnifiedTracker:
    """
    Real-time emotion tracker that captures:
    - Facial emotions from webcam
    - Voice emotions from microphone
    Simultaneously in real-time
    """
    
    def __init__(self):
        print("Initializing Live Unified Tracker...")
        
        # Initialize emotion detectors
        self.face_bot = EmotionBot()
        self.voice_bot = VoiceEmotionBot(sample_rate=16000)  # Lower sample rate for real-time
        
        # Audio recording settings
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 16000
        self.audio_chunk = 1024
        self.audio_frames = []
        self.is_recording = False
        
        # Data storage
        self.emotion_data = []
        
        print("✅ Facial emotion detector ready")
        print("✅ Voice emotion detector ready")
        print("✅ Live unified tracker initialized")
    
    def record_audio_thread(self, duration):
        """
        Background thread for recording audio from microphone
        """
        try:
            p = pyaudio.PyAudio()
            
            # Open microphone stream
            stream = p.open(
                format=self.audio_format,
                channels=self.audio_channels,
                rate=self.audio_rate,
                input=True,
                frames_per_buffer=self.audio_chunk
            )
            
            print("🎤 Microphone recording started")
            
            # Record audio
            start_time = time.time()
            while self.is_recording and (time.time() - start_time) < duration:
                data = stream.read(self.audio_chunk, exception_on_overflow=False)
                self.audio_frames.append(data)
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            print("🎤 Microphone recording stopped")
            
        except Exception as e:
            print(f"⚠️  Audio recording error: {e}")
            self.is_recording = False
    
    def save_audio_to_file(self, filename="temp_live_audio.wav"):
        """
        Save recorded audio to WAV file for analysis
        """
        if not self.audio_frames:
            return None
        
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.audio_channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
            wf.setframerate(self.audio_rate)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            return filename
        except Exception as e:
            print(f"⚠️  Error saving audio: {e}")
            return None
    
    def process_live(self, duration_seconds=30, sample_rate=0.5):
        """
        Capture facial + voice emotions simultaneously in real-time
        
        Args:
            duration_seconds: How long to record (seconds)
            sample_rate: Time between emotion samples (seconds)
        
        Returns:
            DataFrame with synchronized facial + voice emotion data
        """
        print(f"\n{'='*70}")
        print(f"🎬 LIVE UNIFIED EMOTION TRACKING")
        print(f"{'='*70}")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Sample rate: {sample_rate} seconds")
        print(f"\n📸 Webcam will show your face")
        print(f"🎤 Microphone will capture your voice")
        print(f"\nPress 'q' to stop early")
        print(f"{'='*70}\n")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not access webcam")
            return pd.DataFrame()
        
        # Start audio recording in background thread
        self.is_recording = True
        self.audio_frames = []
        audio_thread = threading.Thread(
            target=self.record_audio_thread, 
            args=(duration_seconds,)
        )
        audio_thread.start()
        
        # Track timing
        start_time = time.time()
        last_sample_time = start_time
        frame_count = 0
        
        print("Recording started... Speak and show expressions!")
        
        # Main capture loop
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if duration exceeded
            if elapsed >= duration_seconds:
                break
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process facial emotions at sample rate
            if current_time - last_sample_time >= sample_rate:
                try:
                    # Detect emotions using EmotionBot's detector
                    if self.face_bot.use_fer:
                        emotions = self.face_bot.detector.detect_emotions(frame)
                    else:
                        emotions = self.face_bot.detect_emotions_fallback(frame)
                    
                    if emotions:
                        face = emotions[0]
                        emotion_scores = face['emotions']
                        
                        # Calculate emotional dimensions
                        dimensions = self.face_bot.calculate_enhanced_dimensions(emotion_scores)
                        
                        # Store facial emotion data
                        entry = {
                            'timestamp': datetime.now(),
                            'time_seconds': elapsed,
                            'facial_angry': emotion_scores.get('angry', 0),
                            'facial_disgust': emotion_scores.get('disgust', 0),
                            'facial_fear': emotion_scores.get('fear', 0),
                            'facial_happy': emotion_scores.get('happy', 0),
                            'facial_sad': emotion_scores.get('sad', 0),
                            'facial_surprise': emotion_scores.get('surprise', 0),
                            'facial_neutral': emotion_scores.get('neutral', 0),
                            'facial_arousal': dimensions.get('arousal', 0),
                            'facial_valence': dimensions.get('valence', 0),
                            'facial_quadrant': self.face_bot.get_emotion_quadrant(
                                dimensions.get('arousal', 0),
                                dimensions.get('valence', 0)
                            )
                        }
                        self.emotion_data.append(entry)
                        
                        # Display current emotion
                        dominant = max(emotion_scores, key=emotion_scores.get)
                        print(f"[{elapsed:5.1f}s] Facial: {dominant:8s} ({emotion_scores[dominant]:.2f}) | "
                              f"A:{dimensions.get('arousal', 0):5.2f} V:{dimensions.get('valence', 0):5.2f}")
                    
                    last_sample_time = current_time
                    
                except Exception as e:
                    print(f"⚠️  Error analyzing frame: {e}")
            
            # Display frame with emotion overlay
            display_frame = frame.copy()
            
            # Add status text
            cv2.putText(display_frame, f"Recording: {int(elapsed)}s / {duration_seconds}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to stop", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show if currently sampling
            if current_time - last_sample_time < 0.1:
                cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            
            cv2.imshow('Live Unified Emotion Tracker (Facial + Voice)', display_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Stopped by user")
                break
        
        # Cleanup
        self.is_recording = False
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📸 Captured {len(self.emotion_data)} facial emotion samples")
        
        # Wait for audio thread to finish
        print("⏳ Waiting for audio recording to finish...")
        audio_thread.join(timeout=5)
        
        # Analyze recorded audio
        print("\n🎤 Analyzing voice emotions...")
        voice_df = pd.DataFrame()
        
        if self.audio_frames:
            audio_file = self.save_audio_to_file()
            if audio_file:
                try:
                    # Process the recorded audio
                    voice_df = self.voice_bot.process_audio_file(audio_file, segment_duration=sample_rate)
                    print(f"✅ Analyzed {len(voice_df)} voice emotion samples")
                    
                    # Clean up temp file
                    import os
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                        
                except Exception as e:
                    print(f"⚠️  Error analyzing voice: {e}")
        else:
            print("⚠️  No audio data captured")
        
        # Synchronize facial and voice data
        print("\n🔄 Synchronizing facial and voice data...")
        synchronized_df = self.synchronize_data(voice_df)
        
        print(f"✅ Processing complete: {len(synchronized_df)} synchronized samples")
        
        return synchronized_df
    
    def synchronize_data(self, voice_df):
        """
        Synchronize facial emotion data with voice emotion data
        """
        if not self.emotion_data:
            return pd.DataFrame()
        
        # Convert facial data to dataframe
        facial_df = pd.DataFrame(self.emotion_data)
        
        if voice_df.empty:
            # No voice data - return facial only
            print("⚠️  No voice data to synchronize - returning facial data only")
            return facial_df
        
        # Merge by time
        synchronized_data = []
        
        for _, facial_row in facial_df.iterrows():
            facial_time = facial_row['time_seconds']
            
            # Find nearest voice sample
            if 'time_seconds' in voice_df.columns:
                time_diffs = (voice_df['time_seconds'] - facial_time).abs()
                nearest_idx = time_diffs.idxmin()
                voice_row = voice_df.iloc[nearest_idx]
                
                # Combine data
                entry = facial_row.to_dict()
                
                # Add voice emotions
                voice_emotions = ['voice_angry', 'voice_disgust', 'voice_fear',
                                 'voice_happy', 'voice_sad', 'voice_surprise', 'voice_neutral']
                for emotion in voice_emotions:
                    if emotion in voice_row:
                        entry[emotion] = voice_row[emotion]
                
                # Add voice features
                voice_features = ['pitch_mean', 'volume_mean', 'speech_rate', 
                                 'voice_arousal', 'voice_valence', 'voice_intensity']
                for feature in voice_features:
                    if feature in voice_row:
                        entry[feature] = voice_row[feature]
                
                # Calculate combined metrics
                entry['combined_arousal'] = (facial_row['facial_arousal'] + 
                                            voice_row.get('voice_arousal', 0)) / 2
                entry['combined_valence'] = (facial_row['facial_valence'] + 
                                            voice_row.get('voice_valence', 0)) / 2
                
                synchronized_data.append(entry)
            else:
                synchronized_data.append(facial_row.to_dict())
        
        return pd.DataFrame(synchronized_data)
    
    def get_summary(self, df):
        """
        Generate summary of captured emotions
        """
        if df.empty:
            return {}
        
        summary = {
            'total_samples': len(df),
            'duration': df['time_seconds'].max() - df['time_seconds'].min() if 'time_seconds' in df.columns else 0
        }
        
        # Facial emotion averages
        facial_emotions = ['facial_angry', 'facial_happy', 'facial_sad', 'facial_fear']
        for emotion in facial_emotions:
            if emotion in df.columns:
                summary[emotion] = df[emotion].mean()
        
        # Voice emotion averages
        voice_emotions = ['voice_angry', 'voice_happy', 'voice_sad', 'voice_fear']
        for emotion in voice_emotions:
            if emotion in df.columns:
                summary[emotion] = df[emotion].mean()
        
        # Combined metrics
        if 'combined_arousal' in df.columns:
            summary['combined_arousal'] = df['combined_arousal'].mean()
        if 'combined_valence' in df.columns:
            summary['combined_valence'] = df['combined_valence'].mean()
        
        return summary

if __name__ == "__main__":
    # Quick test
    tracker = LiveUnifiedTracker()
    
    print("\nStarting 10-second test...")
    print("Speak and make facial expressions!")
    
    df = tracker.process_live(duration_seconds=10, sample_rate=1.0)
    
    if not df.empty:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        summary = tracker.get_summary(df)
        for key, value in summary.items():
            print(f"{key:20s}: {value}")
        
        # Save results
        df.to_csv("live_test.csv", index=False)
        print("\n✅ Saved to live_test.csv")
