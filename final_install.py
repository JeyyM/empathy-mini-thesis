import subprocess
import sys
import os

def install_package(package):
    """Install a single package with error handling"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def test_imports():
    """Test all critical imports"""
    print("\n" + "="*50)
    print("TESTING IMPORTS")
    print("="*50)
    
    # Test basic packages
    packages_to_test = [
        ("cv2", "OpenCV"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("numpy", "NumPy")
    ]
    
    for module, name in packages_to_test:
        try:
            __import__(module)
            print(f"✓ {name} import successful")
        except ImportError:
            print(f"✗ {name} import failed")
    
    # Test moviepy specifically
    try:
        import moviepy.editor
        print("✓ moviepy.editor import successful")
        moviepy_ok = True
    except ImportError as e:
        print(f"✗ moviepy.editor import failed: {e}")
        moviepy_ok = False
    
    # Test FER
    try:
        from fer import FER
        if moviepy_ok:
            detector = FER()
            print("✓ FER initialization successful")
        else:
            print("⚠ FER available but moviepy issue may cause problems")
    except ImportError:
        print("✗ FER import failed")
    except Exception as e:
        print(f"✗ FER initialization failed: {e}")
    
    # Test DeepFace
    try:
        from deepface import DeepFace
        print("✓ DeepFace import successful")
    except ImportError:
        print("✗ DeepFace import failed")
    
    # Test librosa (for voice emotion)
    try:
        import librosa
        print("✓ librosa import successful")
    except ImportError:
        print("✗ librosa import failed")
    
    # Test audonnx (for ML voice model)
    try:
        import audonnx
        print("✓ audonnx import successful (ML voice model)")
    except ImportError:
        print("⚠ audonnx import failed (ML voice model unavailable, will use rule-based)")
    
    # Test TextBlob
    try:
        from textblob import TextBlob
        print("✓ TextBlob import successful")
    except ImportError:
        print("✗ TextBlob import failed (optional)")

def fix_moviepy():
    """Attempt to fix moviepy installation issues"""
    print("\n" + "="*50)
    print("FIXING MOVIEPY ISSUES")
    print("="*50)
    
    commands = [
        # Uninstall existing moviepy
        [sys.executable, "-m", "pip", "uninstall", "moviepy", "-y"],
        
        # Install dependencies first
        [sys.executable, "-m", "pip", "install", "imageio", "imageio-ffmpeg"],
        [sys.executable, "-m", "pip", "install", "decorator", "tqdm", "requests"],
        
        # Try specific moviepy version
        [sys.executable, "-m", "pip", "install", "moviepy==1.0.3"],
    ]
    
    for cmd in commands:
        try:
            print(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            print("✓ Success")
        except subprocess.CalledProcessError:
            print("✗ Failed (continuing anyway)")
        except FileNotFoundError:
            print(f"✗ Command not found: {cmd[0]}")

def main():
    """Main installation routine"""
    print("="*60)
    print("EMOTION TRACKER - COMPLETE INSTALLATION SCRIPT")
    print("="*60)
    print("\nThis will install packages for:")
    print("  📸 Facial emotion detection (FER, DeepFace)")
    print("  🎤 Voice emotion detection (librosa, audonnx)")
    print("  🤖 ML models (Wav2vec2 transformer)")
    print("  📊 Data processing and visualization")
    
    # Core packages required for basic functionality
    core_packages = [
        "opencv-python",
        "pandas", 
        "matplotlib",
        "numpy",
        "scipy"
    ]
    
    # Emotion detection packages
    emotion_packages = [
        "fer",
        "deepface",
        "tensorflow"
    ]
    
    # Voice emotion packages
    voice_packages = [
        "librosa",
        "soundfile",
        "audonnx",  # ML model support
        "audiofile",
        "opensmile"  # Acoustic features
    ]
    
    # MoviePy and dependencies
    moviepy_packages = [
        "imageio",
        "imageio-ffmpeg", 
        "decorator",
        "tqdm",
        "requests",
        "moviepy==1.0.3"
    ]
    
    # Optional enhancement packages
    optional_packages = [
        "textblob",
        "Pillow"
    ]
    
    print("\nStep 1: Installing core packages...")
    print("-" * 40)
    for package in core_packages:
        install_package(package)
    
    print("\nStep 2: Installing emotion detection packages...")
    print("-" * 40)
    for package in emotion_packages:
        install_package(package)
    
    print("\nStep 3: Installing voice emotion packages (including ML)...")
    print("-" * 40)
    for package in voice_packages:
        install_package(package)
    
    print("\nStep 4: Installing MoviePy with dependencies...")
    print("-" * 40)
    for package in moviepy_packages:
        install_package(package)
    
    print("\nStep 5: Installing optional packages...")
    print("-" * 40)
    for package in optional_packages:
        install_package(package)
    
    # Try to fix moviepy issues
    fix_moviepy()
    
    # Test all imports
    test_imports()
    
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE!")
    print("="*60)
    
    # Check if ML model needs downloading
    if os.path.exists("emotion_model"):
        print("\n✅ ML voice model already installed")
    else:
        print("\n⚠️  ML voice model not yet downloaded")
        print("   Run: python download_emotion_model.py")
        print("   (Required for ML-enhanced voice emotion detection)")
    
    print("\n📋 What you can run now:")
    print("\n  1. ML-Enhanced Pipeline (Recommended):")
    print("     python new_main.py")
    print("     - Uses Wav2vec2 ML model for voice (93%+ accuracy)")
    print("     - Better angry/happy discrimination")
    print("     - Outputs have '_ml' prefix")
    
    print("\n  2. Original Pipeline:")
    print("     python main.py")
    print("     - Uses rule-based voice emotion detection")
    print("     - Faster but less accurate")
    
    print("\n  3. Download ML Model (if needed):")
    print("     python download_emotion_model.py")
    print("     - One-time setup for ML voice detection")
    
    print("\n📚 Documentation:")
    print("   - QUICKSTART_ML.md         (How to use ML system)")
    print("   - ML_VOICE_IMPLEMENTATION.md (Technical details)")
    print("   - PROJECT_COMPLETE.md      (Project overview)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
