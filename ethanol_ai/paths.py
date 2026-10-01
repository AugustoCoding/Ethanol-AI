"""Caminhos do repositório, independentes da pasta de onde o código é executado."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
EXPERIMENTAL = DATA / "experimental"
SYNTHETIC = DATA / "synthetic"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
