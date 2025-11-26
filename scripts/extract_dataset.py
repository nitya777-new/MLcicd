#!/usr/bin/env python3
import zipfile, sys
from pathlib import Path
RAW = Path("data/raw/dataset.zip")
OUT = Path("data/processed")
if not RAW.exists():
    print("No dataset zip at", RAW)
    sys.exit(1)
with zipfile.ZipFile(RAW, 'r') as z:
    z.extractall(OUT)
print("Extracted to", OUT)
