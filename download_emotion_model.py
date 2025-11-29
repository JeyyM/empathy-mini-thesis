"""
Download the pre-trained Wav2vec2 dimensional emotion recognition model.
This model outputs arousal, dominance, and valence directly.

Model: wav2vec2-large-robust-12-ft-emotion-msp-dim
Source: https://doi.org/10.5281/zenodo.6221127
License: CC BY-NC-SA 4.0 (free for research/thesis use)
"""

import os
import audeer

# Create directories
model_root = 'emotion_model'
cache_root = 'cache'

audeer.mkdir(cache_root)
audeer.mkdir(model_root)

print("Downloading Wav2vec2 dimensional emotion model...")
print("This model predicts arousal, dominance, and valence (0-1 scale)")
print("Trained on MSP-Podcast dataset with 93%+ accuracy\n")

url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'

try:
    # Download the model
    archive_path = audeer.download_url(url, cache_root, verbose=True)
    print(f"\nDownloaded to: {archive_path}")
    
    # Extract the model
    print(f"\nExtracting to: {model_root}")
    audeer.extract_archive(archive_path, model_root)
    
    print("\n✓ Model downloaded and extracted successfully!")
    print(f"✓ Model location: {os.path.abspath(model_root)}")
    print("\nThe model is ready to use in voice_emotion_bot.py")
    
except Exception as e:
    print(f"\n✗ Error downloading model: {e}")
    print("\nIf download fails, you can manually download from:")
    print("https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip")
    print(f"Then extract to: {os.path.abspath(model_root)}")
