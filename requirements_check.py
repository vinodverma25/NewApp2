
#!/usr/bin/env python3
"""
Requirements checker for YouTube Shorts Generator
"""

import importlib
import sys
import os

REQUIRED_PACKAGES = [
    'flask',
    'flask_sqlalchemy',
    'psycopg2',
    'google-auth',
    'google-auth-oauthlib',
    'google-auth-httplib2',
    'google-api-python-client',
    'yt-dlp',
    'opencv-python',
    'numpy',
    'pillow',
    'psutil'
]

OPTIONAL_PACKAGES = [
    'google.generativeai',
    'moviepy',
    'whisper'
]

def check_package(package_name):
    """Check if a package is available"""
    try:
        importlib.import_module(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def check_requirements():
    """Check all requirements"""
    print("=== YouTube Shorts Generator Requirements Check ===\n")
    
    missing_required = []
    missing_optional = []
    
    print("Required packages:")
    for package in REQUIRED_PACKAGES:
        if check_package(package):
            print(f"✓ {package}")
        else:
            print(f"✗ {package} (MISSING)")
            missing_required.append(package)
    
    print("\nOptional packages:")
    for package in OPTIONAL_PACKAGES:
        if check_package(package):
            print(f"✓ {package}")
        else:
            print(f"- {package} (optional, not installed)")
            missing_optional.append(package)
    
    print("\nEnvironment variables:")
    env_vars = [
        'DATABASE_URL',
        'GOOGLE_API_KEY',
        'YOUTUBE_CLIENT_ID',
        'YOUTUBE_CLIENT_SECRET'
    ]
    
    for var in env_vars:
        if os.environ.get(var):
            print(f"✓ {var}")
        else:
            print(f"✗ {var} (not set)")
    
    print("\n" + "="*50)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Run: pip install " + " ".join(missing_required))
        return False
    else:
        print("\n✅ All required packages are installed!")
        if missing_optional:
            print(f"Note: Optional packages not installed: {', '.join(missing_optional)}")
        return True

if __name__ == "__main__":
    success = check_requirements()
    sys.exit(0 if success else 1)
