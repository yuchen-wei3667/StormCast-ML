import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print("PyTorch imported successfully!")
except ImportError as e:
    print(f"Failed to import PyTorch: {e}")

try:
    import json
    print("JSON imported successfully!")
except ImportError as e:
    print(f"Failed to import json: {e}")
