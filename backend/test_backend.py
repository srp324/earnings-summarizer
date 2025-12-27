"""Quick test script to verify backend can start."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.config import get_settings
    from app.main import app
    print("✅ Backend imports successful")
    print(f"✅ Python: {sys.executable}")
    print(f"✅ Python version: {sys.version}")
    
    settings = get_settings()
    print(f"✅ Settings loaded")
    print(f"   - Host: {settings.host}")
    print(f"   - Port: {settings.port}")
    print(f"   - OpenAI API Key: {'Set' if settings.openai_api_key else 'NOT SET'}")
    
    print("\n✅ Backend is ready to start!")
    print("   Run: python run_debug.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

