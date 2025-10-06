import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import warnings

# Suppress specific warnings from FER/Keras
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", message=".*The structure of.*doesn't match.*")

# Try importing FER with fallback
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError as e:
    print(f"FER import error: {e}")
    print("Trying alternative approach...")
    FER_AVAILABLE = False

# Try importing additional libraries for arousal/valence
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Try importing additional libraries for enhanced emotion analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import matplotlib.patches as patches
    MATPLOTLIB_PATCHES_AVAILABLE = True
except ImportError:
    MATPLOTLIB_PATCHES_AVAILABLE = False

class EmotionBot:
    def __init__(self):
        if FER_AVAILABLE:
            try:
                # Test if moviepy.editor is available before initializing FER
                import moviepy.editor
                
                # Initialize FER with warning suppression
                import os
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
                
                print("Initializing FER detector...")
                self.detector = FER(mtcnn=True)
                self.use_fer = True
                print("Using FER emotion detector (warnings suppressed)")
            except ImportError as e:
                print(f"moviepy.editor not available: {e}")
                self.use_fer = False
            except Exception as e:
                print(f"FER initialization failed: {e}")
                self.use_fer = False
        else:
            self.use_fer = False
            
        if not self.use_fer:
            # Fallback to OpenCV face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Using OpenCV face detection with simulated emotions")
            print("Note: For real emotion detection, fix the moviepy installation")
            
        # Try to initialize DeepFace for arousal/valence
        self.use_deepface = False
        if DEEPFACE_AVAILABLE:
            try:
                print("DeepFace available for extended emotion analysis")
                self.use_deepface = True
            except Exception as e:
                print(f"DeepFace initialization failed: {e}")
                self.use_deepface = False
        
        # Enhanced emotion-to-dimension mappings for circumplex model
        self.emotion_dimensions = {
            'happy': {'arousal': 0.8, 'valence': 0.9, 'excitement': 0.9, 'positivity': 0.9},
            'surprise': {'arousal': 0.9, 'valence': 0.1, 'excitement': 0.8, 'positivity': 0.3},
            'angry': {'arousal': 0.8, 'valence': -0.8, 'excitement': 0.7, 'positivity': -0.8},
            'fear': {'arousal': 0.7, 'valence': -0.6, 'excitement': 0.6, 'positivity': -0.7},
            'sad': {'arousal': -0.6, 'valence': -0.7, 'excitement': -0.5, 'positivity': -0.8},
            'disgust': {'arousal': 0.3, 'valence': -0.8, 'excitement': 0.2, 'positivity': -0.7},
            'neutral': {'arousal': 0.0, 'valence': 0.0, 'excitement': 0.0, 'positivity': 0.0}
        }
        
        # Quadrant mappings for circumplex model
        self.quadrant_labels = {
            (1, 1): "Excited",      # High arousal, positive valence
            (1, -1): "Frustrated",  # High arousal, negative valence
            (-1, 1): "Relaxed",     # Low arousal, positive valence
            (-1, -1): "Calming"     # Low arousal, negative valence
        }
        
        self.emotion_data = []
        
    def detect_emotions_fallback(self, frame):
        """Fallback emotion detection using face detection only"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # More realistic emotion simulation based on time
            # This creates a pattern that changes over time for demo purposes
            time_factor = len(self.emotion_data) * 0.1
            
            emotions = {
                'angry': max(0, 0.1 + 0.2 * np.sin(time_factor * 0.5)),
                'disgust': max(0, 0.05 + 0.1 * np.sin(time_factor * 0.3)),
                'fear': max(0, 0.1 + 0.15 * np.sin(time_factor * 0.7)),
                'happy': max(0, 0.3 + 0.4 * np.sin(time_factor * 0.2)),
                'sad': max(0, 0.15 + 0.2 * np.sin(time_factor * 0.4)),
                'surprise': max(0, 0.1 + 0.15 * np.sin(time_factor * 0.9)),
                'neutral': max(0, 0.2 + 0.3 * np.cos(time_factor * 0.1))
            }
            
            # Normalize to sum to 1
            total = sum(emotions.values())
            emotions = {k: v/total for k, v in emotions.items()}
            
            x, y, w, h = faces[0]
            return [{
                'box': (x, y, w, h),
                'emotions': emotions
            }]
        
        return []
        
    def calculate_enhanced_dimensions(self, emotion_scores):
        """Calculate enhanced psychological dimensions including excitement and positivity"""
        weighted_arousal = 0
        weighted_valence = 0
        weighted_excitement = 0
        weighted_positivity = 0
        total_intensity = 0
        
        for emotion, score in emotion_scores.items():
            if emotion in self.emotion_dimensions:
                dims = self.emotion_dimensions[emotion]
                weighted_arousal += score * dims['arousal']
                weighted_valence += score * dims['valence']
                weighted_excitement += score * dims['excitement']
                weighted_positivity += score * dims['positivity']
                total_intensity += score
        
        # Intensity is the sum of all non-neutral emotions
        intensity = total_intensity - emotion_scores.get('neutral', 0)
        
        # Calculate calmness as inverse of excitement
        calmness = -weighted_excitement
        
        # Calculate negativity as inverse of positivity
        negativity = -weighted_positivity
        
        return {
            'arousal': max(-1, min(1, weighted_arousal)),
            'valence': max(-1, min(1, weighted_valence)),
            'intensity': max(0, min(1, intensity)),
            'excitement': max(-1, min(1, weighted_excitement)),
            'calmness': max(-1, min(1, calmness)),
            'positivity': max(-1, min(1, weighted_positivity)),
            'negativity': max(-1, min(1, negativity))
        }
    
    def get_emotion_quadrant(self, arousal, valence):
        """Determine which quadrant of the circumplex model the emotion falls into"""
        arousal_sign = 1 if arousal >= 0 else -1
        valence_sign = 1 if valence >= 0 else -1
        return self.quadrant_labels.get((arousal_sign, valence_sign), "Neutral")
    
    def analyze_with_deepface(self, frame):
        """Analyze emotions using DeepFace for additional metrics"""
        try:
            # DeepFace analyze returns emotion probabilities and other metrics
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(result, list):
                result = result[0]
            
            # Extract emotion scores
            emotions = result.get('emotion', {})
            
            # Convert to our format (lowercase keys)
            emotion_scores = {k.lower(): v/100.0 for k, v in emotions.items()}
            
            # Map some DeepFace emotions to our format
            if 'happiness' in emotion_scores:
                emotion_scores['happy'] = emotion_scores.pop('happiness')
            if 'sadness' in emotion_scores:
                emotion_scores['sad'] = emotion_scores.pop('sadness')
            
            return emotion_scores
            
        except Exception as e:
            print(f"DeepFace analysis failed: {e}")
            return None

    def draw_emotion_wheel(self, frame, arousal, valence, x_offset=20, y_offset=20, radius=80):
        """Draw a real-time emotion wheel on the frame"""
        center_x = x_offset + radius
        center_y = y_offset + radius
        
        # Draw outer circle
        cv2.circle(frame, (center_x, center_y), radius, (100, 100, 100), 2)
        
        # Draw quadrant lines
        cv2.line(frame, (center_x - radius, center_y), (center_x + radius, center_y), (100, 100, 100), 1)
        cv2.line(frame, (center_x, center_y - radius), (center_x, center_y + radius), (100, 100, 100), 1)
        
        # Draw quadrant labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        
        # Excited (top right)
        cv2.putText(frame, "Excited", (center_x + 10, center_y - 40), font, font_scale, (0, 255, 255), 1)
        # Frustrated (top left)
        cv2.putText(frame, "Frustrated", (center_x - 70, center_y - 40), font, font_scale, (0, 100, 255), 1)
        # Relaxed (bottom right)
        cv2.putText(frame, "Relaxed", (center_x + 10, center_y + 50), font, font_scale, (100, 255, 100), 1)
        # Calming (bottom left)
        cv2.putText(frame, "Calming", (center_x - 60, center_y + 50), font, font_scale, (150, 150, 255), 1)
        
        # Draw axis labels
        cv2.putText(frame, "Positive", (center_x + radius + 5, center_y), font, font_scale, (100, 255, 100), 1)
        cv2.putText(frame, "Negative", (center_x - radius - 60, center_y), font, font_scale, (100, 100, 255), 1)
        cv2.putText(frame, "Exciting", (center_x - 20, center_y - radius - 10), font, font_scale, (255, 255, 100), 1)
        cv2.putText(frame, "Calming", (center_x - 20, center_y + radius + 20), font, font_scale, (255, 150, 150), 1)
        
        # Calculate position of current emotion
        emotion_x = int(center_x + valence * radius * 0.8)
        emotion_y = int(center_y - arousal * radius * 0.8)  # Negative because y increases downward
        
        # Draw current emotion point
        cv2.circle(frame, (emotion_x, emotion_y), 8, (0, 0, 255), -1)
        cv2.circle(frame, (emotion_x, emotion_y), 12, (255, 255, 255), 2)
        
        return frame

    def process_video(self, video_path, sample_rate=1):
        """Process video and extract emotions frame by frame"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return pd.DataFrame()
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * sample_rate)  # Sample every N seconds
        
        frame_count = 0
        start_time = datetime.now()
        
        print(f"Video FPS: {fps}, Frame interval: {frame_interval}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Detect emotions in current frame
                if self.use_fer:
                    emotions = self.detector.detect_emotions(frame)
                else:
                    emotions = self.detect_emotions_fallback(frame)
                
                if emotions:
                    # Get the most prominent face
                    face = emotions[0]
                    emotion_scores = face['emotions']
                    
                    # Try to get additional analysis from DeepFace
                    if self.use_deepface:
                        deepface_emotions = self.analyze_with_deepface(frame)
                        if deepface_emotions:
                            # Blend FER and DeepFace results (weighted average)
                            for emotion in emotion_scores:
                                if emotion in deepface_emotions:
                                    emotion_scores[emotion] = (emotion_scores[emotion] * 0.7 + 
                                                             deepface_emotions[emotion] * 0.3)
                    
                    # Calculate enhanced dimensions
                    dimensions = self.calculate_enhanced_dimensions(emotion_scores)
                    quadrant = self.get_emotion_quadrant(dimensions['arousal'], dimensions['valence'])
                    
                    # Calculate timestamp
                    timestamp = start_time + timedelta(seconds=frame_count/fps)
                    
                    # Store data
                    data_point = {
                        'timestamp': timestamp,
                        'frame': frame_count,
                        'time_seconds': frame_count/fps,
                        'quadrant': quadrant,
                        **dimensions,
                        **emotion_scores
                    }
                    self.emotion_data.append(data_point)
                    
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                    print(f"Frame {frame_count} ({frame_count/fps:.1f}s): {dominant_emotion} ({emotion_scores[dominant_emotion]:.3f}) | "
                          f"A:{dimensions['arousal']:.2f} V:{dimensions['valence']:.2f} | {quadrant}")
        
            frame_count += 1
            
        cap.release()
        print(f"Processed {frame_count} total frames")
        return self.get_dataframe()
    
    def process_webcam(self, duration_seconds=60, sample_rate=0.5):
        """Process live webcam feed for specified duration"""
        cap = cv2.VideoCapture(0)
        
        # Set optimal resolution for face detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam resolution: {actual_width}x{actual_height}")
        
        start_time = time.time()
        last_sample = 0
        
        # Window settings
        window_name = 'Live Emotion Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set window size to maintain aspect ratio
        display_width = 1024
        display_height = int(display_width * actual_height / actual_width)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        # Text settings - scale based on resolution
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, actual_width / 1000.0)  # Scale font with resolution
        thickness = max(1, int(actual_width / 640))   # Scale thickness
        
        current_emotion = "Detecting..."
        current_confidence = 0.0
        current_arousal = 0.0
        current_valence = 0.0
        current_intensity = 0.0
        current_excitement = 0.0
        current_calmness = 0.0
        current_positivity = 0.0
        current_negativity = 0.0
        current_quadrant = "Neutral"
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get current frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if current_time - last_sample >= sample_rate:
                # Detect emotions
                if self.use_fer:
                    emotions = self.detector.detect_emotions(frame)
                else:
                    emotions = self.detect_emotions_fallback(frame)
                
                if emotions:
                    face = emotions[0]
                    emotion_scores = face['emotions']
                    
                    # Try to get additional analysis from DeepFace
                    if self.use_deepface:
                        deepface_emotions = self.analyze_with_deepface(frame)
                        if deepface_emotions:
                            # Blend results
                            for emotion in emotion_scores:
                                if emotion in deepface_emotions:
                                    emotion_scores[emotion] = (emotion_scores[emotion] * 0.7 + 
                                                             deepface_emotions[emotion] * 0.3)
                    
                    # Calculate enhanced dimensions
                    dimensions = self.calculate_enhanced_dimensions(emotion_scores)
                    current_quadrant = self.get_emotion_quadrant(dimensions['arousal'], dimensions['valence'])
                    
                    timestamp = datetime.now()
                    data_point = {
                        'timestamp': timestamp,
                        'elapsed_seconds': elapsed_time,
                        'quadrant': current_quadrant,
                        **dimensions,
                        **emotion_scores
                    }
                    self.emotion_data.append(data_point)
                    
                    # Update display values
                    current_emotion = max(emotion_scores, key=emotion_scores.get)
                    current_confidence = emotion_scores[current_emotion]
                    current_arousal = dimensions['arousal']
                    current_valence = dimensions['valence']
                    current_intensity = dimensions['intensity']
                    current_excitement = dimensions['excitement']
                    current_calmness = dimensions['calmness']
                    current_positivity = dimensions['positivity']
                    current_negativity = dimensions['negativity']
                    
                    # Draw bounding box around face with better scaling
                    x, y, w, h = face['box']
                    box_thickness = max(2, int(frame_width / 320))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), box_thickness)
                    
                    # Draw emotion label above face with better sizing
                    emotion_text = f"{current_emotion.capitalize()}"
                    confidence_text = f"({current_confidence:.2f})"
                    
                    # Calculate text size and position
                    (text_width, text_height), baseline = cv2.getTextSize(emotion_text, font, font_scale, thickness)
                    (conf_width, conf_height), _ = cv2.getTextSize(confidence_text, font, font_scale*0.8, thickness-1)
                    
                    # Position text above face with padding
                    text_x = max(5, x)
                    text_y = max(text_height + 15, y - 15)
                    
                    # Draw text background
                    bg_width = text_width + conf_width + 15
                    bg_height = text_height + 10
                    cv2.rectangle(frame, (text_x - 5, text_y - bg_height), 
                                 (text_x + bg_width, text_y + 5), (0, 255, 0), -1)
                    
                    # Draw text
                    cv2.putText(frame, emotion_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
                    cv2.putText(frame, confidence_text, (text_x + text_width + 5, text_y), 
                               font, font_scale*0.8, (0, 0, 0), max(1, thickness-1))
                
                last_sample = current_time
            
            # Draw emotion wheel
            frame = self.draw_emotion_wheel(frame, current_arousal, current_valence, 
                                          frame_width - 200, 20, 80)
            
            # Draw overlay information with enhanced metrics
            overlay_height = int(frame_height * 0.35)  # Increased for more metrics
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width - 220, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            
            # Calculate responsive text positioning
            line_height = int(overlay_height / 8)
            margin = int(frame_width * 0.02)
            
            # Display enhanced status
            status_text = f"Emotion: {current_emotion.upper()}"
            confidence_display = f"Confidence: {current_confidence:.1%}"
            quadrant_text = f"State: {current_quadrant.upper()}"
            
            # Core dimensions
            arousal_text = f"Arousal: {current_arousal:+.2f}"
            valence_text = f"Valence: {current_valence:+.2f}"
            intensity_text = f"Intensity: {current_intensity:.2f}"
            
            # Enhanced dimensions
            excitement_text = f"Excitement: {current_excitement:+.2f}"
            calmness_text = f"Calmness: {current_calmness:+.2f}"
            positivity_text = f"Positivity: {current_positivity:+.2f}"
            negativity_text = f"Negativity: {current_negativity:+.2f}"
            
            time_text = f"Time: {elapsed_time:.1f}s / {duration_seconds}s"
            samples_text = f"Samples: {len(self.emotion_data)}"
            
            # Display text in rows
            cv2.putText(frame, status_text, (margin, line_height), font, font_scale*0.9, (0, 255, 0), thickness)
            cv2.putText(frame, confidence_display, (margin, line_height*2), font, font_scale*0.9, (255, 255, 0), thickness)
            cv2.putText(frame, quadrant_text, (margin, line_height*3), font, font_scale*0.9, (255, 100, 255), thickness)
            
            # Core dimensions row
            dim_x1 = int(frame_width * 0.25)
            dim_x2 = int(frame_width * 0.5)
            cv2.putText(frame, arousal_text, (margin, line_height*4), font, font_scale*0.7, (255, 100, 100), thickness-1)
            cv2.putText(frame, valence_text, (dim_x1, line_height*4), font, font_scale*0.7, (100, 255, 100), thickness-1)
            cv2.putText(frame, intensity_text, (dim_x2, line_height*4), font, font_scale*0.7, (100, 100, 255), thickness-1)
            
            # Enhanced dimensions rows
            cv2.putText(frame, excitement_text, (margin, line_height*5), font, font_scale*0.6, (255, 255, 100), thickness-1)
            cv2.putText(frame, calmness_text, (dim_x1, line_height*5), font, font_scale*0.6, (150, 255, 150), thickness-1)
            cv2.putText(frame, positivity_text, (margin, line_height*6), font, font_scale*0.6, (100, 255, 255), thickness-1)
            cv2.putText(frame, negativity_text, (dim_x1, line_height*6), font, font_scale*0.6, (255, 150, 150), thickness-1)
            
            # Timing info
            cv2.putText(frame, time_text, (margin, line_height*7), font, font_scale*0.6, (255, 255, 255), max(1, thickness-1))
            cv2.putText(frame, samples_text, (dim_x1, line_height*7), font, font_scale*0.6, (255, 255, 255), max(1, thickness-1))
            
            # Draw progress bar with responsive sizing
            bar_width = int(frame_width * 0.3)  # 30% of frame width
            bar_height = max(8, int(frame_height / 90))
            bar_x = frame_width - bar_width - margin
            bar_y = margin
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Progress bar
            progress = elapsed_time / duration_seconds
            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
            
            # Instructions at bottom
            instruction_text = "Press 'q' to quit early"
            cv2.putText(frame, instruction_text, (frame_width - 250, frame_height - 20), 
                       font, font_scale*0.6, (255, 255, 255), max(1, thickness-1))
            
            try:
                cv2.imshow(window_name, frame)
            except Exception as e:
                print(f"Display error: {e}")
                break
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nStopped early at {elapsed_time:.1f} seconds")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nWebcam session completed: {len(self.emotion_data)} samples collected")
        return self.get_dataframe()
    
    def get_dataframe(self):
        """Convert emotion data to pandas DataFrame"""
        return pd.DataFrame(self.emotion_data)
    
    def plot_emotions(self, save_path=None):
        """Plot emotion values and enhanced psychological dimensions over time"""
        if not self.emotion_data:
            print("No emotion data to plot")
            return
        
        df = self.get_dataframe()
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Create three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 16))
        
        # Use time in seconds for x-axis
        if 'time_seconds' in df.columns:
            x_axis = df['time_seconds']
            x_label = 'Time (seconds)'
        elif 'elapsed_seconds' in df.columns:
            x_axis = df['elapsed_seconds']
            x_label = 'Time (seconds)'
        else:
            x_axis = df.index
            x_label = 'Sample Number'
        
        # Plot emotions
        for emotion in emotions:
            if emotion in df.columns:
                ax1.plot(x_axis, df[emotion], label=emotion.capitalize(), marker='o', markersize=2)
        
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Emotion Intensity')
        ax1.set_title('Emotion Recognition Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot core psychological dimensions
        core_dims = ['arousal', 'valence', 'intensity']
        colors = ['red', 'green', 'blue']
        for dim, color in zip(core_dims, colors):
            if dim in df.columns:
                ax2.plot(x_axis, df[dim], label=dim.capitalize(), color=color, linewidth=2)
        
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Dimension Value')
        ax2.set_title('Core Psychological Dimensions Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot enhanced dimensions
        enhanced_dims = ['excitement', 'calmness', 'positivity', 'negativity']
        colors2 = ['orange', 'purple', 'cyan', 'pink']
        for dim, color in zip(enhanced_dims, colors2):
            if dim in df.columns:
                ax3.plot(x_axis, df[dim], label=dim.capitalize(), color=color, linewidth=2)
        
        ax3.set_xlabel(x_label)
        ax3.set_ylabel('Dimension Value')
        ax3.set_title('Enhanced Psychological Dimensions Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_emotion_summary(self):
        """Get statistical summary of emotions"""
        if not self.emotion_data:
            return None
        
        df = self.get_dataframe()
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        summary = {}
        for emotion in emotions:
            if emotion in df.columns:
                summary[emotion] = {
                    'mean': df[emotion].mean(),
                    'max': df[emotion].max(),
                    'min': df[emotion].min(),
                    'std': df[emotion].std()
                }
        
        return summary
    
    def clear_data(self):
        """Clear stored emotion data"""
        self.emotion_data = []
