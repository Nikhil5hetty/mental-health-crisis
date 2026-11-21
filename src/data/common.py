import json
import math
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / 'data/raw'
# src/data/common.py -> repository root is parents[2].

def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)

def files(root, pattern):
    return sorted(p for p in Path(root).rglob(pattern) if '__MACOSX' not in p.parts)

def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else float('nan')
    except (TypeError, ValueError):
        return float('nan')
