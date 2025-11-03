#!/usr/bin/env python3

# Test script to verify quadrant mapping logic
emotion_dimensions = {
    'happy': {'arousal': 0.5, 'valence': 0.8, 'excitement': 0.6, 'positivity': 0.8},
    'surprise': {'arousal': 0.9, 'valence': 0.1, 'excitement': 0.8, 'positivity': 0.3},
    'angry': {'arousal': 0.8, 'valence': -0.8, 'excitement': 0.7, 'positivity': -0.8},
    'fear': {'arousal': 0.7, 'valence': -0.6, 'excitement': 0.6, 'positivity': -0.7},
    'sad': {'arousal': -0.8, 'valence': -0.7, 'excitement': -0.7, 'positivity': -0.8},
    'disgust': {'arousal': -0.4, 'valence': -0.8, 'excitement': -0.5, 'positivity': -0.7},
    'neutral': {'arousal': -0.6, 'valence': 0.3, 'excitement': -0.7, 'positivity': 0.2}
}

quadrant_labels = {
    (1, 1): "Excited",      # High arousal, positive valence
    (1, -1): "Agitated",    # High arousal, negative valence  
    (-1, 1): "Calm",        # Low arousal, positive valence
    (-1, -1): "Depressed"   # Low arousal, negative valence
}

def get_emotion_quadrant(arousal, valence):
    """Determine which quadrant of the circumplex model the emotion falls into"""
    arousal_sign = 1 if arousal >= 0 else -1
    valence_sign = 1 if valence >= 0 else -1
    return quadrant_labels.get((arousal_sign, valence_sign), "Neutral")

print("Emotion mappings to quadrants:")
print("=" * 50)
for emotion, dims in emotion_dimensions.items():
    arousal = dims['arousal']
    valence = dims['valence']
    quadrant = get_emotion_quadrant(arousal, valence)
    print(f"{emotion:>8}: arousal={arousal:+.1f}, valence={valence:+.1f} -> {quadrant}")

print("\nTest cases:")
print("=" * 50)
test_cases = [
    (-0.5, 0.3, 'Low arousal, positive valence -> Should be Calm'),
    (-0.3, 0.1, 'Low arousal, slightly positive -> Should be Calm'),
    (-0.4, -0.2, 'Low arousal, negative valence -> Should be Depressed'),
    (0.8, 0.7, 'High arousal, positive -> Should be Excited'),
    (0.6, -0.5, 'High arousal, negative -> Should be Agitated')
]

for arousal, valence, description in test_cases:
    quadrant = get_emotion_quadrant(arousal, valence)
    print(f'{description}: {quadrant}')