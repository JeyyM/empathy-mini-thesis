"""
Test script for voice emotion detection
"""
from voice_emotion_bot import VoiceEmotionBot
import os

def test_voice_emotion_bot():
    """Test the voice emotion detection system"""
    print("🧪 Testing Voice Emotion Bot...")
    
    # Initialize bot
    try:
        bot = VoiceEmotionBot()
        print("✅ Voice bot initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing voice bot: {e}")
        return False
    
    # Test with a sample video that has audio
    test_files = ["testvid.mp4", "happy.mp4", "depressed.mp4"]
    
    for video_file in test_files:
        if os.path.exists(video_file):
            print(f"\n📹 Testing with {video_file}...")
            
            # Extract audio for testing
            try:
                import moviepy.editor as mp
                video_clip = mp.VideoFileClip(video_file)
                
                if video_clip.audio is not None:
                    print(f"✅ {video_file} has audio track")
                    
                    # Test audio extraction
                    temp_audio = f"test_audio_{video_file}.wav"
                    video_clip.audio.write_audiofile(temp_audio, fps=22050, verbose=False, logger=None)
                    
                    if os.path.exists(temp_audio):
                        print(f"✅ Audio extracted to {temp_audio}")
                        
                        # Test voice emotion detection
                        df = bot.process_audio_file(temp_audio, segment_duration=2.0)
                        
                        if not df.empty:
                            print(f"✅ Voice emotions detected: {len(df)} segments")
                            
                            # Show sample results
                            if len(df) > 0:
                                sample = df.iloc[0]
                                print(f"   Sample results:")
                                print(f"   - Voice Arousal: {sample.get('voice_arousal', 0):.3f}")
                                print(f"   - Voice Valence: {sample.get('voice_valence', 0):.3f}")
                                print(f"   - Voice Happy: {sample.get('voice_happy', 0):.3f}")
                                print(f"   - Pitch: {sample.get('pitch_mean', 0):.1f} Hz")
                        else:
                            print("⚠️  No voice emotions detected")
                        
                        # Clean up
                        os.remove(temp_audio)
                        print(f"🧹 Cleaned up {temp_audio}")
                    
                    video_clip.close()
                    return True
                else:
                    print(f"⚠️  {video_file} has no audio track")
                    video_clip.close()
            
            except Exception as e:
                print(f"❌ Error processing {video_file}: {e}")
                
        else:
            print(f"📁 {video_file} not found")
    
    print("❌ No suitable video files found for testing")
    return False

if __name__ == "__main__":
    test_voice_emotion_bot()