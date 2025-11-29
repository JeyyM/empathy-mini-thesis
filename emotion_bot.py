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
            'happy': {'arousal': 0.5, 'valence': 0.8, 'excitement': 0.6, 'positivity': 0.8},
            'surprise': {'arousal': 0.9, 'valence': 0.1, 'excitement': 0.8, 'positivity': 0.3},
            'angry': {'arousal': 0.8, 'valence': -0.8, 'excitement': 0.7, 'positivity': -0.8},
            'fear': {'arousal': 0.7, 'valence': -0.6, 'excitement': 0.6, 'positivity': -0.7},
            'sad': {'arousal': -0.8, 'valence': -0.7, 'excitement': -0.7, 'positivity': -0.8},
            'disgust': {'arousal': -0.4, 'valence': -0.8, 'excitement': -0.5, 'positivity': -0.7},
            'neutral': {'arousal': -0.6, 'valence': 0.3, 'excitement': -0.7, 'positivity': 0.2}
        }
        
        # Quadrant mappings for circumplex model
        self.quadrant_labels = {
            (1, 1): "Excited",      # High arousal, positive valence
            (1, -1): "Stressed",    # High arousal, negative valence  
            (-1, 1): "Calm",        # Low arousal, positive valence
            (-1, -1): "Tired"       # Low arousal, negative valence
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

    def draw_emotion_wheel(self, frame, arousal, valence, x_offset=20, y_offset=20, radius=100):
        """Draw an enhanced, user-friendly emotion wheel on the frame"""
        center_x = x_offset + radius
        center_y = y_offset + radius
        
        # Draw gradient background for quadrants
        overlay = frame.copy()
        
        # Create quadrant colors (subtle backgrounds)
        # Excited quadrant (top right) - warm yellow
        pts_excited = np.array([[center_x, center_y], [center_x + radius, center_y], 
                               [center_x + radius, center_y - radius], [center_x, center_y - radius]], np.int32)
        cv2.fillPoly(overlay, [pts_excited], (100, 200, 255))
        
        # Agitated quadrant (top left) - orange/red
        pts_agitated = np.array([[center_x, center_y], [center_x - radius, center_y], 
                                [center_x - radius, center_y - radius], [center_x, center_y - radius]], np.int32)
        cv2.fillPoly(overlay, [pts_agitated], (0, 100, 255))
        
        # Calm quadrant (bottom right) - cool green
        pts_calm = np.array([[center_x, center_y], [center_x + radius, center_y], 
                            [center_x + radius, center_y + radius], [center_x, center_y + radius]], np.int32)
        cv2.fillPoly(overlay, [pts_calm], (100, 255, 150))
        
        # Depressed quadrant (bottom left) - cool blue
        pts_depressed = np.array([[center_x, center_y], [center_x - radius, center_y], 
                                 [center_x - radius, center_y + radius], [center_x, center_y + radius]], np.int32)
        cv2.fillPoly(overlay, [pts_depressed], (200, 150, 100))
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw outer circle with thicker border
        cv2.circle(frame, (center_x, center_y), radius, (70, 70, 70), 3)
        
        # Draw concentric circles for intensity levels
        for i in range(1, 4):
            inner_radius = int(radius * i / 4)
            cv2.circle(frame, (center_x, center_y), inner_radius, (120, 120, 120), 1)
        
        # Draw quadrant lines
        cv2.line(frame, (center_x - radius, center_y), (center_x + radius, center_y), (70, 70, 70), 2)
        cv2.line(frame, (center_x, center_y - radius), (center_x, center_y + radius), (70, 70, 70), 2)
        
        # Enhanced quadrant labels with better positioning and colors
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Excited quadrant
        cv2.putText(frame, "EXCITED", (center_x + 15, center_y - 45), font, font_scale, (0, 150, 255), thickness)
        cv2.putText(frame, "Energized", (center_x + 15, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)
        
        # Agitated quadrant  
        cv2.putText(frame, "STRESSED", (center_x - 85, center_y - 45), font, font_scale, (0, 50, 255), thickness)
        cv2.putText(frame, "Anxious", (center_x - 85, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 50, 255), 1)
        
        # Calm quadrant
        cv2.putText(frame, "PEACEFUL", (center_x + 15, center_y + 45), font, font_scale, (50, 200, 50), thickness)
        cv2.putText(frame, "Relaxed", (center_x + 15, center_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 200, 50), 1)
        
        # Depressed quadrant
        cv2.putText(frame, "TIRED", (center_x - 85, center_y + 45), font, font_scale, (150, 100, 50), thickness)
        cv2.putText(frame, "Low mood", (center_x - 85, center_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 100, 50), 1)
        
        # Draw clearer axis labels
        cv2.putText(frame, "GOOD MOOD", (center_x + radius + 10, center_y - 5), font, 0.5, (50, 200, 50), 2)
        cv2.putText(frame, "BAD MOOD", (center_x - radius - 80, center_y - 5), font, 0.5, (50, 50, 200), 2)
        cv2.putText(frame, "HIGH ENERGY", (center_x - 45, center_y - radius - 15), font, 0.5, (255, 200, 0), 2)
        cv2.putText(frame, "LOW ENERGY", (center_x - 40, center_y + radius + 25), font, 0.5, (100, 100, 200), 2)
        
        # Calculate position of current emotion with bounds checking
        emotion_x = int(center_x + max(-radius*0.9, min(radius*0.9, valence * radius * 0.8)))
        emotion_y = int(center_y - max(-radius*0.9, min(radius*0.9, arousal * radius * 0.8)))
        
        # Draw emotion trail (last few positions for smoothness)
        if hasattr(self, 'emotion_positions'):
            if len(self.emotion_positions) > 10:
                self.emotion_positions.pop(0)
        else:
            self.emotion_positions = []
        
        self.emotion_positions.append((emotion_x, emotion_y))
        
        # Draw trail
        for i, pos in enumerate(self.emotion_positions[:-1]):
            alpha = i / len(self.emotion_positions)
            trail_color = (int(100 * alpha), int(100 * alpha), int(255 * alpha))
            cv2.circle(frame, pos, 3, trail_color, -1)
        
        # Draw current emotion point with pulsing effect
        pulse_radius = 12 + int(3 * np.sin(len(self.emotion_positions) * 0.3))
        cv2.circle(frame, (emotion_x, emotion_y), pulse_radius, (255, 255, 255), 2)
        cv2.circle(frame, (emotion_x, emotion_y), 8, (0, 0, 255), -1)
        
        # Add intensity indicator
        intensity = np.sqrt(arousal**2 + valence**2)
        intensity_text = f"Intensity: {intensity:.2f}"
        cv2.putText(frame, intensity_text, (x_offset, y_offset + 2*radius + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add current quadrant text
        current_quadrant = self.get_emotion_quadrant(arousal, valence)
        cv2.putText(frame, f"Current: {current_quadrant.upper()}", (x_offset, y_offset + 2*radius + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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
    
    def plot_heatmap(self, save_path=None, bins=20):
        """Create a heatmap showing emotion distribution in arousal-valence space"""
        if not self.emotion_data:
            print("No emotion data available for heatmap")
            return
        
        df = self.get_dataframe()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Emotion Distribution Heatmaps', fontsize=16, fontweight='bold')
        
        # 1. Arousal-Valence density heatmap
        ax1 = axes[0, 0]
        arousal_vals = df['arousal'].values
        valence_vals = df['valence'].values
        
        h1 = ax1.hist2d(valence_vals, arousal_vals, bins=bins, cmap='YlOrRd', alpha=0.8)
        ax1.set_xlabel('Valence (Negative ← → Positive)')
        ax1.set_ylabel('Arousal (Calm ← → Exciting)')
        ax1.set_title('Arousal-Valence Density')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax1.text(0.7, 0.7, 'Excited', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(-0.7, 0.7, 'Agitated', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(0.7, -0.7, 'Calm', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(-0.7, -0.7, 'Depressed', ha='center', va='center', fontsize=10, fontweight='bold')
        
        plt.colorbar(h1[3], ax=ax1, label='Frequency')
        
        # 2. Intensity heatmap
        ax2 = axes[0, 1]
        intensity_vals = df['intensity'].values
        h2 = ax2.hist2d(valence_vals, arousal_vals, bins=bins, weights=intensity_vals, cmap='plasma', alpha=0.8)
        ax2.set_xlabel('Valence (Negative ← → Positive)')
        ax2.set_ylabel('Arousal (Calm ← → Exciting)')
        ax2.set_title('Emotional Intensity Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='white', linestyle='--', alpha=0.7)
        ax2.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        plt.colorbar(h2[3], ax=ax2, label='Average Intensity')
        
        # 3. Dominant emotion heatmap
        ax3 = axes[1, 0]
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_colors = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 
            'sad': 4, 'surprise': 5, 'neutral': 6
        }
        
        dominant_emotions = []
        for _, row in df.iterrows():
            emotion_scores = {emotion: row[emotion] for emotion in emotions if emotion in row}
            dominant = max(emotion_scores, key=emotion_scores.get)
            dominant_emotions.append(emotion_colors[dominant])
        
        scatter = ax3.scatter(valence_vals, arousal_vals, c=dominant_emotions, 
                            cmap='tab10', alpha=0.7, s=30)
        ax3.set_xlabel('Valence (Negative ← → Positive)')
        ax3.set_ylabel('Arousal (Calm ← → Exciting)')
        ax3.set_title('Dominant Emotion by Position')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Create custom legend for emotions
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=plt.cm.tab10(emotion_colors[emotion]/9), 
                                    markersize=8, label=emotion.capitalize()) 
                         for emotion in emotions]
        ax3.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # 4. Time evolution heatmap
        ax4 = axes[1, 1]
        if 'elapsed_seconds' in df.columns:
            time_vals = df['elapsed_seconds'].values
            h4 = ax4.hist2d(time_vals, arousal_vals, bins=bins, cmap='viridis', alpha=0.8)
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Arousal (Calm ← → Exciting)')
            ax4.set_title('Arousal Evolution Over Time')
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
        
        plt.show()
    
    def generate_layperson_report(self, save_path=None):
        """Generate a simple, easy-to-understand emotion report for laypeople"""
        if not self.emotion_data:
            print("No emotion data available for report")
            return
        
        df = self.get_dataframe()
        
        # Determine column naming convention
        if 'arousal' in df.columns:
            arousal_col = 'arousal'
            valence_col = 'valence'
            quadrant_col = 'quadrant'
        elif 'facial_arousal' in df.columns:
            arousal_col = 'facial_arousal'
            valence_col = 'facial_valence'
            quadrant_col = 'facial_quadrant'
        else:
            print("⚠️ No arousal/valence data found in dataframe")
            return
        
        # Create a clean, simple report figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Your Emotion Report - Easy to Understand', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Overall mood pie chart (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        emotions = ['happy', 'neutral', 'sad', 'angry', 'fear', 'surprise', 'disgust']
        emotion_totals = {emotion: df[emotion].sum() for emotion in emotions if emotion in df.columns}
        
        # Simplify to main emotions
        main_emotions = {'Positive': emotion_totals.get('happy', 0) + emotion_totals.get('surprise', 0),
                        'Neutral': emotion_totals.get('neutral', 0),
                        'Worried': emotion_totals.get('fear', 0) + emotion_totals.get('sad', 0),
                        'Upset': emotion_totals.get('angry', 0) + emotion_totals.get('disgust', 0)}
        
        colors = ['#90EE90', '#D3D3D3', '#FFB6C1', '#FFA07A']
        wedges, texts, autotexts = ax1.pie(main_emotions.values(), labels=main_emotions.keys(), 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Overall Mood Distribution', fontsize=14, fontweight='bold')
        
        # 2. Energy levels over time (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        time_data = np.arange(len(df))
        energy_levels = ['Very Low' if x < -0.5 else 'Low' if x < -0.2 else 'Medium' if x < 0.3 else 'High' if x < 0.6 else 'Very High' 
                        for x in df[arousal_col]]
        energy_colors = {'Very Low': '#4169E1', 'Low': '#87CEEB', 'Medium': '#FFFF00', 'High': '#FFA500', 'Very High': '#FF4500'}
        
        for i, (level, color) in enumerate(energy_colors.items()):
            mask = [x == level for x in energy_levels]
            if any(mask):
                ax2.scatter([j for j, m in enumerate(mask) if m], 
                           [df[arousal_col].iloc[j] for j, m in enumerate(mask) if m], 
                           c=color, label=level, s=30, alpha=0.7)
        
        ax2.set_xlabel('Time →')
        ax2.set_ylabel('Energy Level')
        ax2.set_title('Your Energy Throughout Session', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Mood levels over time (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        mood_levels = ['Very Negative' if x < -0.5 else 'Negative' if x < -0.2 else 'Neutral' if x < 0.2 else 'Positive' if x < 0.5 else 'Very Positive' 
                      for x in df[valence_col]]
        mood_colors = {'Very Negative': '#8B0000', 'Negative': '#CD5C5C', 'Neutral': '#D3D3D3', 'Positive': '#90EE90', 'Very Positive': '#00FF00'}
        
        for level, color in mood_colors.items():
            mask = [x == level for x in mood_levels]
            if any(mask):
                ax3.scatter([j for j, m in enumerate(mask) if m], 
                           [df[valence_col].iloc[j] for j, m in enumerate(mask) if m], 
                           c=color, label=level, s=30, alpha=0.7)
        
        ax3.set_xlabel('Time →')
        ax3.set_ylabel('Mood Level')
        ax3.set_title('Your Mood Throughout Session', fontsize=14, fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Emotional states distribution (middle row, full width)
        ax4 = fig.add_subplot(gs[1, :])
        quadrant_counts = df[quadrant_col].value_counts()
        quadrant_colors = {'Excited': '#FFD700', 'Stressed': '#FF6347', 'Calm': '#90EE90', 'Tired': '#87CEEB'}
        
        bars = ax4.bar(quadrant_counts.index, quadrant_counts.values, 
                      color=[quadrant_colors.get(q, '#D3D3D3') for q in quadrant_counts.index])
        ax4.set_title('Time Spent in Different Emotional States', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Number of Moments')
        
        # Add percentage labels on bars
        total_moments = len(df)
        for bar, count in zip(bars, quadrant_counts.values):
            percentage = (count / total_moments) * 100
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Key insights text (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        
        # Calculate insights
        dominant_emotion = max(main_emotions, key=main_emotions.get)
        avg_energy = df[arousal_col].mean()
        avg_mood = df[valence_col].mean()
        most_common_state = quadrant_counts.index[0]
        
        energy_desc = "high" if avg_energy > 0.2 else "low" if avg_energy < -0.2 else "moderate"
        mood_desc = "positive" if avg_mood > 0.2 else "negative" if avg_mood < -0.2 else "neutral"
        
        insights_text = f"""KEY INSIGHTS:
        
• Most common feeling: {dominant_emotion}
• Average energy: {energy_desc.capitalize()}
• Average mood: {mood_desc.capitalize()}
• Spent most time: {most_common_state}

SUMMARY:
You showed mostly {dominant_emotion.lower()} 
emotions with {energy_desc} energy levels 
and a generally {mood_desc} mood."""
        
        ax5.text(0.05, 0.95, insights_text, transform=ax5.transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # 6. Recommendations (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # Generate simple recommendations
        recommendations = []
        if avg_energy < -0.3:
            recommendations.append("• Try energizing activities")
            recommendations.append("• Get enough sleep")
        elif avg_energy > 0.5:
            recommendations.append("• Practice relaxation techniques")
            recommendations.append("• Take breaks when needed")
        
        if avg_mood < -0.3:
            recommendations.append("• Engage in mood-boosting activities")
            recommendations.append("• Connect with supportive people")
        elif avg_mood > 0.3:
            recommendations.append("• Keep doing what makes you happy!")
        
        if not recommendations:
            recommendations = ["• You seem well-balanced!", "• Continue your current approach"]
        
        rec_text = "SUGGESTIONS:\n\n" + "\n".join(recommendations)
        ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        # 7. Emotion timeline (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        # Simple timeline showing emotional journey
        timeline_colors = [quadrant_colors.get(q, '#D3D3D3') for q in df[quadrant_col]]
        ax7.scatter(range(len(df)), [0]*len(df), c=timeline_colors, s=50, alpha=0.7)
        ax7.set_xlim(-1, len(df))
        ax7.set_ylim(-0.5, 0.5)
        ax7.set_xlabel('Time →')
        ax7.set_title('Your Emotional Journey', fontsize=12, fontweight='bold')
        ax7.set_yticks([])
        
        # Add legend for timeline
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=8, label=state) 
                         for state, color in quadrant_colors.items()]
        ax7.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layperson report saved to {save_path}")
        
        plt.show()
    
    def plot_circle_movement_heatmap(self, save_path=None, grid_size=50):
        """Create a heatmap showing movement patterns within the emotion circle"""
        if not self.emotion_data:
            print("No emotion data available for movement heatmap")
            return
        
        df = self.get_dataframe()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Emotion Circle Movement Analysis', fontsize=16, fontweight='bold')
        
        # Extract arousal and valence data (handle both naming conventions)
        if 'arousal' in df.columns:
            arousal_col = 'arousal'
            valence_col = 'valence'
            quadrant_col = 'quadrant'
            arousal_vals = df['arousal'].values
            valence_vals = df['valence'].values
        elif 'facial_arousal' in df.columns:
            arousal_col = 'facial_arousal'
            valence_col = 'facial_valence'
            quadrant_col = 'facial_quadrant'
            arousal_vals = df['facial_arousal'].values
            valence_vals = df['facial_valence'].values
        else:
            print("⚠️ No arousal/valence data found in dataframe")
            return
        
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
        ax1.text(0.7, -0.7, 'PEACEFUL\nRelaxed', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax1.text(-0.7, -0.7, 'TIRED\nLow mood', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Add axis labels
        ax1.set_xlabel('MOOD: Bad ← → Good', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ENERGY: Low ← → High', fontsize=12, fontweight='bold')
        ax1.set_title('Where You Spend Time\n(Movement Density)', fontsize=14, fontweight='bold')
        
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
                        distance = np.sqrt(dx**2 + dy**2)
                        weight = np.exp(-distance**2 / (2 * sigma**2))
                        gaussian_grid[ny, nx] += weight
        
        # Apply circular mask
        gaussian_masked = np.where(circle_mask, gaussian_grid, np.nan)
        
        im2 = ax2.imshow(gaussian_masked, extent=[-1, 1, -1, 1], origin='lower', cmap='plasma', alpha=0.8)
        
        # Draw movement path
        ax2.plot(valence_vals, arousal_vals, 'white', linewidth=2, alpha=0.8, label='Emotion Path')
        ax2.scatter(valence_vals[0], arousal_vals[0], color='lime', s=100, marker='o', 
                   label='Start', edgecolor='black', linewidth=2, zorder=5)
        ax2.scatter(valence_vals[-1], arousal_vals[-1], color='red', s=100, marker='s', 
                   label='End', edgecolor='black', linewidth=2, zorder=5)
        
        # Draw circle boundary
        circle2 = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=2)
        ax2.add_patch(circle2)
        
        # Draw quadrant lines
        ax2.axhline(y=0, color='white', linestyle='-', alpha=0.7, linewidth=1)
        ax2.axvline(x=0, color='white', linestyle='-', alpha=0.7, linewidth=1)
        
        ax2.set_xlabel('MOOD: Bad ← → Good', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ENERGY: Low ← → High', fontsize=12, fontweight='bold')
        ax2.set_title('Your Emotional Journey\n(Movement Path)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Movement Intensity', fontsize=10)
        
        ax2.legend(loc='upper left', bbox_to_anchor=(0, 1))
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Circle movement heatmap saved to {save_path}")
        
        plt.show()
        
        # Print movement statistics
        print(f"\n=== Movement Analysis ===")
        print(f"Total emotion samples: {len(df)}")
        
        # Calculate movement distance
        if len(valence_vals) > 1:
            distances = []
            for i in range(1, len(valence_vals)):
                dist = np.sqrt((valence_vals[i] - valence_vals[i-1])**2 + 
                              (arousal_vals[i] - arousal_vals[i-1])**2)
                distances.append(dist)
            
            total_distance = sum(distances)
            avg_distance = np.mean(distances)
            
            print(f"Total movement distance: {total_distance:.2f}")
            print(f"Average step distance: {avg_distance:.3f}")
            print(f"Movement activity: {'High' if avg_distance > 0.1 else 'Moderate' if avg_distance > 0.05 else 'Low'}")
        
        # Quadrant time analysis
        if quadrant_col in df.columns:
            quadrant_time = df[quadrant_col].value_counts()
            print(f"\nTime in each emotional state:")
            for quadrant, count in quadrant_time.items():
                percentage = (count / len(df)) * 100
                print(f"  {quadrant}: {count} moments ({percentage:.1f}%)")
        
        # Most extreme positions
        max_positive = df.loc[df[valence_col].idxmax()]
        max_negative = df.loc[df[valence_col].idxmin()]
        max_energy = df.loc[df[arousal_col].idxmax()]
        min_energy = df.loc[df[arousal_col].idxmin()]
        
        print(f"\nExtreme positions:")
        print(f"  Most positive mood: {max_positive[valence_col]:.2f} ({max_positive.get(quadrant_col, 'Unknown')})")
        print(f"  Most negative mood: {max_negative[valence_col]:.2f} ({max_negative.get(quadrant_col, 'Unknown')})")
        print(f"  Highest energy: {max_energy[arousal_col]:.2f} ({max_energy.get(quadrant_col, 'Unknown')})")
        print(f"  Lowest energy: {min_energy[arousal_col]:.2f} ({min_energy.get(quadrant_col, 'Unknown')})")
    
    def clear_data(self):
        """Clear stored emotion data"""
        self.emotion_data = []
