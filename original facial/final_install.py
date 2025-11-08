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
    
    # Core packages required for basic functionality
    core_packages = [
        "opencv-python",
        "pandas", 
        "matplotlib",
        "numpy"
    ]
    
    # Emotion detection packages
    emotion_packages = [
        "fer",
        "deepface",
        "tensorflow"
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
    
    print("\nStep 3: Installing MoviePy with dependencies...")
    print("-" * 40)
    for package in moviepy_packages:
        install_package(package)
    
    print("\nStep 4: Installing optional packages...")
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
    print("You can now run:")
    print("  python main.py")
    print("\nFor webcam mode, enter: camera")
    print("For video file mode, enter: your_video_filename.mp4")
    print("\nIf you still have issues with FER, the system will")
    print("automatically fall back to OpenCV with simulated emotions.")

if __name__ == "__main__":
    main()
