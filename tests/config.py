# tests/config.py
import sys
from pathlib import Path

# Adds project root to sys.path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))